#pragma once

#include "types.h"
#include <exception>
#include <iostream>
#include <stdlib.h>

namespace NeuralNetwork {

class Matrix {
    private:
    size_nnt rows;
    size_nnt cols;

    real_nnt** data = nullptr;

    public:

        Matrix(real_nnt* arr, int start, int rows, int cols) : rows(rows), cols(cols) {
            data = new real_nnt*[rows];
            int c = 0;
            for (int i = 0; i < rows; i++) {
                data[i] = new real_nnt[cols];
                for (int j = 0; j < cols; j++) {
                    data[i][j] = arr[c++];
                }
            }
        }

    real_nnt** get_raw_ptr() {
        return data;
    }

    real_nnt* unravel() {
        real_nnt *res = new real_nnt[rows * cols];
        int c = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                res[c++] = data[i][j];
            }
        }

        return res;
    }

    size_nnt row_count() const {
        return rows;
    }

    size_nnt col_count() const {
        return cols;
    }

    real_nnt value(int r, int c) const {
        return data[r][c];
    }
    real_nnt& value(int r, int c) {
        return data[r][c];
    }

    void zeroify() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = 0;
            }
        }
    }

    void max_index(int& r, int& c) {
        r = 0;
        c = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (data[i][j] > data[r][c]) {
                    r = i;
                    c = j;
                }
            }
        }
    }

    Matrix() = delete;

    Matrix(size_nnt rows, size_nnt cols) : rows(rows), cols(cols) {
        data = new real_nnt*[rows];
        for (int i = 0; i < rows; i++) {
            data[i] = new real_nnt[cols]; 
            for (int j = 0; j < cols; j++) {
                data[i][j] = 0;
            }
        }
    }

    ~Matrix() {
        if (data != nullptr) {
            for (int i = 0; i < rows; i++) {
                delete[] data[i];
            }
            delete[] data;
        }
        data = nullptr;
    }

    Matrix(Matrix&& move_from) : rows(move_from.rows), cols(move_from.cols) {
        this->data = move_from.data;
        move_from.data = nullptr;
        move_from.rows = 0;
        move_from.cols = 0;
    }
    Matrix& operator=(Matrix&& move_from) noexcept {

        if (this->data != nullptr) {
            for (int i = 0; i < rows; i++) {
                delete[] data[i];
            }
            delete[] data;
        }

        this->data = move_from.data;
        this->rows = move_from.rows;
        this->cols = move_from.cols;

        move_from.data = nullptr;

        return *this;
    }

    void randomize(float start, float end) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = start + (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * (end - start);
            }
        }
    }

    void add(const Matrix& other, real_nnt other_scale_factor, real_nnt this_scale_factor) {
        if (other.cols != cols || other.rows != rows) {
            throw std::exception("Matrix dimensions mismatch");
        }

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = other_scale_factor * other.data[i][j] + data[i][j] * this_scale_factor;
            }
        }
    }

    void add(const Matrix& other) {
        add(other, 1.0, 1.0);
    }

    void subtract(const Matrix& other) {
        add(other, -1.0, 1.0);
    }

    void multiply(float scalar) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] *= scalar;
            }
        }
    }

    Matrix multiply(const Matrix& other) const {
        return multiply(*this, other, 1.0, 1.0);
    }

    void apply_function(activation_func_nnt func) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = func(data[i][j]);
            }
        }
    }

    void multiply_elementwise(const Matrix& other)  {
        if (other.rows != rows || other.cols != cols) {
            throw std::exception("Invalid matrix sizes.");
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] *= other.data[i][j];
            }
        }
    }

    Matrix transpose() const {
        Matrix res = Matrix(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                res.data[j][i] = data[i][j];
            }
        }

        return res;
    }

    static Matrix multiply(const Matrix& a, const Matrix& b, real_nnt a_scalar, real_nnt b_scalar) {
        if (a.cols != b.rows) {
            throw std::exception("Matrix dimensions mismatch");
        }
        Matrix res(a.rows, b.cols);
        for (int i = 0; i < res.rows; i++) {
            for (int j = 0; j < res.cols; j++) {
                res.data[i][j] = 0;
                for (int k = 0; k < a.cols; k++) {
                    res.data[i][j] += (a_scalar * a.data[i][k]) * (b_scalar * b.data[k][j]);
                }
            }
        }

        return res;
    }

    void print(const char* label = nullptr) const {
        if (label != nullptr) {
            std::cout << label << " of " << rows << "x" << cols << std::endl;
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                std::cout << data[i][j] <<  ",\t";
            }
            std::cout << std::endl;
        }
    }

    Matrix clone() const {
        Matrix res(rows, cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                res.data[i][j] = data[i][j];
            }
        }
        return res;
    }

    Matrix(const Matrix& copy) = delete;
    Matrix operator=(const Matrix&) = delete;
};

}