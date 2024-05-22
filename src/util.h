#pragma once

#include <stdio.h>

// Returned array should be deleted.
const char* read_file(const char* path) {
    char buffer[10000];
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
    //    std::cout << str << std::endl;
    return str;
}

void write_file(char* data, int start, int size, const char* path) {
    FILE* file = fopen(path, "w");
    size_t c = 0;
    while (c < size) {
        fputc(data[start + c++], file);
    }

    fclose(file);
}