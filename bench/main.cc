#include <benchmark/benchmark.h>
#include <argparse/argparse.hpp>
#include <string>

#include "../src/conv2d.hh"
#include "../src/model.hh"
#include "helpers.hh"

void bench(benchmark::State &state, std::vector<unsigned int> dims)
{
	auto batch = dims[0];
	auto channel = dims[1];
	auto height = dims[2];
	auto width = dims[3];

	unsigned int nb_filters = state.range(0);
	unsigned int kernel_size = state.range(1);

	torchinfer::Model<float> model;
	model.add(new torchinfer::Inputs<float>("input", {batch, channel, height, width}));
	model.add(new torchinfer::Conv2D<float>("conv2d", torchinfer::get_uniform_tensor<float>({nb_filters, channel, kernel_size, kernel_size}), {1, 1}));
	model.compile();

	auto x = torchinfer::get_uniform_tensor<float>({batch, channel, height, width});

	for (auto _ : state)
		model.layers[0]->forward(x);
}

// test, name of test, dims, nb_filters, kernel_size
BENCHMARK_CAPTURE(bench, small_input, {1, 3, 224, 224})
	->ArgsProduct({{32, 64, 128, 256, 512},
				   {3}})
	->Unit(benchmark::kMillisecond);

BENCHMARK_CAPTURE(bench, medium_input, {32, 3, 224, 224})
	->ArgsProduct({{32, 64, 128, 256, 512},
				   {3}})
	->Unit(benchmark::kMillisecond);

int main(int argc, char *argv[])
{
	benchmark::Initialize(&argc, argv);
	benchmark::RunSpecifiedBenchmarks();
	benchmark::Shutdown();
}