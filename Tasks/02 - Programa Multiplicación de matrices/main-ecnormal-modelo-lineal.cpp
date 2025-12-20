// main-ecnormal-modelo-lineal.cpp
// C++23 + LAPACK (LAPACKE) — Recta de mejor ajuste y = a x + b
//
// Datos (de la imagen):
// (1,2), (2,2), (3,4), (4,5)
//
// Modelo: y ≈ a x + b
// Forma matricial: A θ ≈ y, donde θ = [a, b]^T
//
// A = [ x  1 ]
// y = [ y ]
//
// "Ecuación normal":
//   (A^T A) θ = A^T y
//
// En este programa:
// 1) Construimos A (m×n) y y (m×1)
// 2) Calculamos ATA = A^T A y ATy = A^T y
// 3) Resolvemos (ATA) θ = ATy con LAPACK: LAPACKE_dgesv
//
// Compilar en MSYS2 UCRT64 (OpenBLAS + LAPACKE):
//   g++ -std=c++23 -O2 -Wall -Wextra main-ecnormal-modelo-lineal.cpp -o app2.exe -llapacke -lopenblas
//
// Ejecutar:
//   ./app2.exe

#include <iostream>
#include <iomanip>
#include <vector>
#include <stdexcept>

#include <lapacke.h>  // LAPACKE_dgesv
#include "openblas/cblas.h"    // cblas_dgemm, cblas_dgemv

int main() {
    try {
        // -----------------------------
        // 0) Datos
        // -----------------------------
        // Puntos: (x_i, y_i)
        const std::vector<double> x = {1, 2, 3, 4};
        const std::vector<double> y_rm = {2, 2, 4, 5}; // lo guardamos "como lista" (row-major conceptual)
        const int m = static_cast<int>(x.size());      // número de muestras (filas)
        const int n = 2;                               // número de parámetros: a y b

        // -----------------------------
        // 1) Construir A (m×n) en column-major
        //    A = [x  1]
        //
        // Column-major significa:
        // A_colmajor(j*m + i) = A(i,j)
        // -----------------------------
        std::vector<double> A(static_cast<size_t>(m) * n);
        for (int i = 0; i < m; ++i) {
            A[0 * m + i] = x[i];  // columna 0: x
            A[1 * m + i] = 1.0;   // columna 1: 1
        }

        // Vector y como columna (m×1). Para BLAS/LAPACK, es un vector.
        std::vector<double> y = y_rm;

        // -----------------------------
        // 2) Calcular ATA = A^T A (n×n)
        //    y ATy = A^T y (n×1)
        // -----------------------------
        std::vector<double> ATA(static_cast<size_t>(n) * n, 0.0); // n×n
        std::vector<double> ATy(static_cast<size_t>(n), 0.0);     // n×1

        // ATA = A^T * A
        // cblas_dgemm: C = alpha * op(A) * op(B) + beta*C
        // Aquí:
        //   op(A) = A^T (n×m)
        //   op(B) = A   (m×n)
        //   C     = ATA (n×n)
        cblas_dgemm(
            CblasColMajor,
            CblasTrans, CblasNoTrans,
            n, n, m,
            1.0,
            A.data(), m,
            A.data(), m,
            0.0,
            ATA.data(), n
        );

        // ATy = A^T * y
        // cblas_dgemv: y_out = alpha*op(A)*x + beta*y_out
        cblas_dgemv(
            CblasColMajor,
            CblasTrans,
            m, n,
            1.0,
            A.data(), m,
            y.data(), 1,
            0.0,
            ATy.data(), 1
        );

        // -----------------------------
        // 3) Resolver (ATA) * theta = ATy
        //    theta = [a, b]^T
        //
        // Usamos LAPACKE_dgesv:
        // - Resuelve sistemas lineales Ax=b con LU + pivoteo parcial.
        // - Modifica la matriz y el vector en sitio.
        // -----------------------------
        std::vector<int> ipiv(n);              // pivotes
        std::vector<double> theta = ATy;       // copiamos ATy porque dgesv sobrescribe b

        int info = LAPACKE_dgesv(
            LAPACK_COL_MAJOR,
            n,        // orden de A (n×n)
            1,        // nrhs: número de columnas de b (aquí 1)
            ATA.data(), n, // matriz A (ATA), leading dimension = n
            ipiv.data(),
            theta.data(), n  // b (ATy), leading dimension = n
        );

        if (info < 0) {
            throw std::runtime_error("LAPACKE_dgesv: argumento ilegal en posición " + std::to_string(-info));
        }
        if (info > 0) {
            throw std::runtime_error("LAPACKE_dgesv: matriz singular; no se pudo resolver (info=" + std::to_string(info) + ")");
        }

        const double a = theta[0];
        const double b = theta[1];

        // -----------------------------
        // 4) Mostrar resultado
        // -----------------------------
        std::cout << std::fixed << std::setprecision(10);
        std::cout << "Recta de mejor ajuste (ecuacion normal):\n";
        std::cout << "y = a*x + b\n";
        std::cout << "a = " << a << "\n";
        std::cout << "b = " << b << "\n\n";

        // Opcional: mostrar predicciones y error cuadrático medio
        double sse = 0.0;
        std::cout << "Puntos y prediccion:\n";
        for (int i = 0; i < m; ++i) {
            double y_hat = a * x[i] + b;
            double err = y_hat - y[i];
            sse += err * err;
            std::cout << "x=" << x[i] << "  y=" << y[i] << "  y_hat=" << y_hat << "  err=" << err << "\n";
        }
        std::cout << "\nSSE = " << sse << "\n";
        std::cout << "MSE = " << (sse / m) << "\n";

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
