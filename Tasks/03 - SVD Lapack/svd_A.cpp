#include <iostream>
#include <iomanip>
#include <cmath>
#include <lapacke.h>

int main() {
    // Matriz A en row-major: fila por fila
    // A = [[1, -0.8],
    //      [0,  1.0]]
    double A[4] = {
        1.0, -0.8,
        0.0,  1.0
    };

    const lapack_int m = 2;
    const lapack_int n = 2;
    const lapack_int lda = n;

    double S[2];    // valores singulares
    double U[4];    // U (2x2)
    double VT[4];   // V^T (2x2)
    double superb[1];

    // 1) SVD con LAPACKE
    lapack_int info = LAPACKE_dgesvd(
        LAPACK_ROW_MAJOR, // formato row-major
        'A',              // calcular U completa
        'A',              // calcular V^T completa
        m, n,
        A, lda,
        S,
        U, m,
        VT, n,
        superb
    );

    if (info != 0) {
        std::cerr << "Error en dgesvd, info = " << info << '\n';
        return 1;
    }

    std::cout << std::fixed << std::setprecision(8);

    std::cout << "Valores singulares S:\n";
    for (int i = 0; i < 2; ++i)
        std::cout << "  S[" << i << "] = " << S[i] << '\n';

    std::cout << "\nMatriz U (2x2):\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j)
            std::cout << std::setw(14) << U[i*n + j] << ' ';
        std::cout << '\n';
    }

    std::cout << "\nMatriz V^T (2x2):\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            std::cout << std::setw(14) << VT[i*n + j] << ' ';
        std::cout << '\n';
    }

    // 2) Reconstruir A_rec = U * Sigma * V^T
    double Sigma[4] = {0.0};
    Sigma[0] = S[0];      // (0,0)
    Sigma[3] = S[1];      // (1,1)

    double US[4] = {0.0};
    double A_rec[4] = {0.0};

    // US = U * Sigma
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 2; ++k) {
                sum += U[i*2 + k] * Sigma[k*2 + j];
            }
            US[i*2 + j] = sum;
        }
    }

    // A_rec = US * V^T
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 2; ++k) {
                sum += US[i*2 + k] * VT[k*2 + j];
            }
            A_rec[i*2 + j] = sum;
        }
    }

    std::cout << "\nA reconstruida (U * Sigma * V^T):\n";
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j)
            std::cout << std::setw(14) << A_rec[i*2 + j] << ' ';
        std::cout << '\n';
    }

    // 3) Error máximo respecto a la A original
    double A_orig[4] = {
        1.0, -0.8,
        0.0,  1.0
    };

    double max_err = 0.0;
    for (int i = 0; i < 4; ++i) {
        double diff = std::fabs(A_rec[i] - A_orig[i]);
        if (diff > max_err) max_err = diff;
    }
    std::cout << "\nError máximo |A_rec - A_orig| = " << max_err << '\n';

    return 0;
}
