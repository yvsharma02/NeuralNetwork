#pragma once

#include <stdio.h>

// Returned array should be deleted.
const char* read_file(const char* path, int buffer_size = 1024 * 1024 * 8) {
    char* buffer = new char[buffer_size];
    int c = 0;

    FILE* file = fopen(path, "r");
    char ch = fgetc(file);
    while (ch != EOF) {
        buffer[c++] = ch;
        ch = fgetc(file);
    }
    fclose(file);
    char* str = new char[c + 1];
    for (int i = 0; i < c; i++) {
        str[i] = buffer[i];
    }
    str[c] = 0;

    delete[] buffer;
    //    std::cout << str << std::endl;
    return str;
}


const char* read_file_bin(const char* path) {
    FILE* f = fopen(path, "rb");
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* string = new char[fsize];
    fread(string, fsize, 1, f);
    fclose(f);

    return string;
}

void write_file(char* data, int start, int size, const char* path) {
    FILE* file = fopen(path, "w");
    size_t c = 0;
    while (c < size) {
        fputc(data[start + c++], file);
    }

    fclose(file);
}

void write_file_bin(char* data, int start, int size, const char* path) {
    FILE* file = fopen(path, "wb");
    size_t c = 0;
    while (c < size) {
        fputc(data[start + c++], file);
    }

    fclose(file);
}


int rand_int(int limit) {
   int x = ((rand() << 32) ^ rand());
   return (x < 0 ? -x : x) % limit;
//    return ((rand() << 2) ^ rand()) % limit;
}

float sigmoid(float f) {
    return 1.0 / (1 + expf(-f));
}

float sigmoid_derivative(float f) {
    return sigmoid(f) * (1 - sigmoid(f));
}