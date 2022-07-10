
#pragma once

#include <gtest/gtest.h>
#include <string>

#include "../../torchinfer/conv2d.hh"

TEST(TestConv2d, print)
{
    testing::internal::CaptureStdout();
    auto layer = torchinfer::Conv2D();
    layer.print();
    std::string result = testing::internal::GetCapturedStdout();
    EXPECT_EQ(result, "Conv2D !\n");
}
