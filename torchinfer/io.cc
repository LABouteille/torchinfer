#include "io.hh"
#include <cstdio>

namespace torchinfer
{
    std::vector<int> read_bin(std::string filename)
    {
        /*
        Read from numpy write_bin().

        file format:
            - row (int)
            - col (int)
            - format [int, float, double] (byte)
            - data
        */

        std::map<char, int> format_to_byte{
            {'c', sizeof(char)},
            {'i', sizeof(int)},
            {'I', sizeof(int)},
            {'f', sizeof(float)},
            {'d', sizeof(double)},
        };

        std::ifstream file(filename, std::ios::binary);

        int row = -1, col = -1;
        file.read(reinterpret_cast<char *>(&row), sizeof(int));
        file.read(reinterpret_cast<char *>(&col), sizeof(int));

        if (row == -1 || col == -1)
            throw std::runtime_error("read_bin: No dimensions (row, col) dumped in binary");

        char format = '\0';
        file.read(reinterpret_cast<char *>(&format), sizeof(char));

        if (format == '\0')
            throw std::runtime_error("read_bin: No format character dumped in binary");

        std::vector<int> vec;
        vec.resize(row * col);

        file.read(reinterpret_cast<char *>(vec.data()), row * col * format_to_byte[format]);

        return vec;
    }

} // namespace torchinfer
