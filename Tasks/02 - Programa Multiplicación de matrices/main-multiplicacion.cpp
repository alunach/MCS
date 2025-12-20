// matmul_dgemm_general.cpp
// Multiplicación general: A (m×n) * B (n×l) = C (m×l) usando DGEMM (BLAS/LAPACK stack)
//
// Entrada (texto) recomendada:
// m n l
// A: m líneas, cada una con n doubles
// B: n líneas, cada una con l doubles
//
// Salida (texto):
// m l
// C: m líneas, cada una con l doubles
//
// Nota: DGEMM (Fortran) espera column-major. Leemos row-major y convertimos.
//
// Compilación (MSYS2 ucrt64 + OpenBLAS):
//   g++ -std=c++23 -O2 -Wall -Wextra matmul_dgemm_general.cpp -o matmul.exe -lopenblas
//
// Ejecución:
//   ./matmul.exe input.txt output.txt

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
// DGEMM: C = alpha*op(A)*op(B) + beta*C
void dgemm_(const char* TRANSA, const char* TRANSB,
            const int* M, const int* N, const int* K,
            const double* ALPHA,
            const double* A, const int* LDA,
            const double* B, const int* LDB,
            const double* BETA,
            double* C, const int* LDC);
}

static std::vector<double> read_matrix_row_major(std::istream& in, int rows, int cols, const std::string& name) {
    std::vector<double> M(static_cast<size_t>(rows) * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (!(in >> M[static_cast<size_t>(i) * cols + j])) {
                throw std::runtime_error("Error leyendo matriz " + name +
                                         " en (" + std::to_string(i) + "," + std::to_string(j) + ").");
            }
        }
    }
    return M;
}

// Row-major (rows×cols) -> Column-major (rows×cols)
static std::vector<double> to_col_major(const std::vector<double>& rm, int rows, int cols) {
    std::vector<double> cm(static_cast<size_t>(rows) * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            cm[static_cast<size_t>(j) * rows + i] = rm[static_cast<size_t>(i) * cols + j];
    return cm;
}

// Column-major (rows×cols) -> Row-major (rows×cols)
static std::vector<double> to_row_major(const std::vector<double>& cm, int rows, int cols) {
    std::vector<double> rm(static_cast<size_t>(rows) * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            rm[static_cast<size_t>(i) * cols + j] = cm[static_cast<size_t>(j) * rows + i];
    return rm;
}

static void write_matrix_row_major(std::ostream& out, const std::vector<double>& rm, int rows, int cols) {
    out << rows << " " << cols << "\n";
    out << std::setprecision(17);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            out << rm[static_cast<size_t>(i) * cols + j];
            if (j + 1 < cols) out << ' ';
        }
        out << '\n';
    }
}

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cerr << "Uso: " << argv[0] << " <input.txt> <output.txt>\n";
            return 1;
        }

        const std::string in_path  = argv[1];
        const std::string out_path = argv[2];

        std::ifstream fin(in_path);
        if (!fin) throw std::runtime_error("No se pudo abrir el archivo de entrada: " + in_path);

        // Leer dimensiones generales
        int m = 0, n = 0, l = 0;
        if (!(fin >> m >> n >> l) || m <= 0 || n <= 0 || l <= 0) {
            throw std::runtime_error("Dimensiones inválidas. Se espera: m n l (enteros positivos).");
        }

        // Leer A (m×n) y B (n×l)
        const auto A_rm = read_matrix_row_major(fin, m, n, "A");
        const auto B_rm = read_matrix_row_major(fin, n, l, "B");

        // Convertir a column-major para DGEMM (Fortran)
        const auto A = to_col_major(A_rm, m, n);
        const auto B = to_col_major(B_rm, n, l);

        // C es (m×l) en column-major
        std::vector<double> C(static_cast<size_t>(m) * l, 0.0);

        // DGEMM parámetros:
        // A: (m×n), B: (n×l), C: (m×l)
        const char trans = 'N';
        const int M = m;
        const int N = l;
        const int K = n;

        // Leading dimensions en column-major = número de filas del arreglo físico
        const int lda = m; // A tiene m filas
        const int ldb = n; // B tiene n filas
        const int ldc = m; // C tiene m filas

        const double alpha = 1.0;
        const double beta  = 0.0;

        dgemm_(&trans, &trans,
               &M, &N, &K,
               &alpha,
               A.data(), &lda,
               B.data(), &ldb,
               &beta,
               C.data(), &ldc);

        // Convertir C a row-major para escribir “normal”
        const auto C_rm = to_row_major(C, m, l);

        std::ofstream fout(out_path);
        if (!fout) throw std::runtime_error("No se pudo abrir el archivo de salida: " + out_path);

        write_matrix_row_major(fout, C_rm, m, l);

        std::cout << "OK: C = A*B con DGEMM. Dimensiones: (" << m << "x" << l << ")\n";
        return 0;

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 2;
    }
}
