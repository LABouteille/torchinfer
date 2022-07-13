#include "io.hh"

namespace torchinfer
{
    std::vector<int> read_bin(std::string filename)
    {
        /*
        Read from numpy write_bin().

        file format:
            - n (int)
            - c (int)
            - h (int)
            - w (int)
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

        int n = -1, c = -1, h = -1, w = -1;
        file.read(reinterpret_cast<char *>(&n), sizeof(int));
        file.read(reinterpret_cast<char *>(&c), sizeof(int));
        file.read(reinterpret_cast<char *>(&h), sizeof(int));
        file.read(reinterpret_cast<char *>(&w), sizeof(int));

        if (n == -1 || c == -1 || h == -1 || w == -1)
            throw std::runtime_error("read_bin: No dimensions (n,c,h,w) dumped in binary");

        char format = '\0';
        file.read(reinterpret_cast<char *>(&format), sizeof(char));

        if (format == '\0')
            throw std::runtime_error("read_bin: No format character dumped in binary");

        std::vector<int> vec(n * c * h * w);

        file.read(reinterpret_cast<char *>(vec.data()), n * c * h * w * format_to_byte[format]);

        return vec;
    }

} // namespace torchinfer
