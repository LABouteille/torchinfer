#include <iostream>
#include <torchinfer/conv2d.hh>
#include <torchinfer/io.hh>

int main(void) {
    std::cout << "Hello World !\n";
    
    auto layer  = torchinfer::Conv2D(); 
    layer.print();

    auto vec = torchinfer::read_bin("../sandbox/array.bin");
    for (auto elt: vec)
        printf("%d\n", elt);

}