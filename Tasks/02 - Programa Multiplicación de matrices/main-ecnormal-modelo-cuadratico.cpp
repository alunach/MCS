// main-ecnormal-modelo-cuadratico.cpp
// Ajuste por mínimos cuadrados de un modelo cuadrático:
//      y ≈ a x^2 + b x + c
// usando LAPACK (LAPACKE_dgels) => resuelve directamente min ||A*theta - y||_2
// (QR), sin formar A^T A (más estable que la ecuación normal).
//
// Datos (según la imagen):
// (0,1.2), (1,2.0), (2,2.9), (3,4.1), (4,5.8), (5,8.2)
//
// Compilar (MSYS2 UCRT64):
//   g++ -std=c++23 -O2 -Wall -Wextra main-ecnormal-modelo-cuadratico.cpp -o app3.exe -llapacke -lopenblas
//
// Ejecutar:
//   ./app3.exe
//
// Salida:
// - coeficientes a,b,c
// - tabla (x, y, y_hat, err)
// - SSE, MSE
// - genera "fit.csv" para graficar en Octave/Excel

#include <lapacke.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    try {
        // -----------------------------
        // 1) Datos
        // -----------------------------
        const std::vector<double> x = {0, 1, 2, 3, 4, 5};
        const std::vector<double> y = {1.2, 2.0, 2.9, 4.1, 5.8, 8.2};

        const lapack_int m = static_cast<lapack_int>(x.size()); // # muestras
        const lapack_int n = 3;                                 // parámetros: a,b,c
        const lapack_int nrhs = 1;                              // una columna (y)

        // -----------------------------
        // 2) Construir matriz de diseño A (m×n)
        //    Para cada fila i:
        //      [ x_i^2,  x_i,  1 ]
        //
        // LAPACKE_dgels espera A en COLUMN-MAJOR:
        //   A(j*m + i) = A(i,j)
        // -----------------------------
        std::vector<double> A(static_cast<size_t>(m) * n);
        for (lapack_int i = 0; i < m; ++i) {
            const double xi = x[static_cast<size_t>(i)];
            A[0 * m + i] = xi * xi; // columna 0: x^2
            A[1 * m + i] = xi;      // columna 1: x
            A[2 * m + i] = 1.0;     // columna 2: 1
        }

        // -----------------------------
        // 3) Vector B para dgels (sobrescrito con la solución)
        //
        // DGELS resuelve min ||A*X - B||_2. Para el caso sobredeterminado (m>=n):
        // - B debe tener tamaño LDB >= max(m,n)
        // - Tras resolver, las primeras n entradas de B contienen X = [a,b,c].
        // - El resto de B (desde n hasta m-1) contiene residuos en forma compacta.
        // -----------------------------
        const lapack_int ldb = std::max(m, n);
        std::vector<double> B(static_cast<size_t>(ldb) * nrhs, 0.0);
        for (lapack_int i = 0; i < m; ++i) {
            B[static_cast<size_t>(i)] = y[static_cast<size_t>(i)];
        }

        // -----------------------------
        // 4) Resolver con DGELS (QR)
        //    trans='N' => usa A tal cual.
        // -----------------------------
        const char trans = 'N';
        const lapack_int info = LAPACKE_dgels(
            LAPACK_COL_MAJOR,
            trans,
            m, n, nrhs,
            A.data(), m,  // lda = m (column-major)
            B.data(), ldb // ldb = max(m,n)
        );

        if (info < 0) {
            throw std::runtime_error("LAPACKE_dgels: argumento ilegal en posicion " + std::to_string(-info));
        }
        if (info > 0) {
            throw std::runtime_error("LAPACKE_dgels: fallo numérico (info=" + std::to_string(info) + ")");
        }

        // Coeficientes (solución) en B[0..n-1]
        const double a = B[0];
        const double b = B[1];
        const double c = B[2];

        // -----------------------------
        // 5) Reporte: modelo y calidad del ajuste
        // -----------------------------
        std::cout << std::fixed << std::setprecision(10);
        std::cout << "Modelo cuadratico (minimos cuadrados con DGELS/QR):\n";
        std::cout << "y = a*x^2 + b*x + c\n";
        std::cout << "a = " << a << "\n";
        std::cout << "b = " << b << "\n";
        std::cout << "c = " << c << "\n\n";

        double sse = 0.0;
        std::cout << "Puntos y prediccion:\n";
        for (size_t i = 0; i < x.size(); ++i) {
            const double xi = x[i];
            const double y_hat = a * xi * xi + b * xi + c;
            const double err = y_hat - y[i];
            sse += err * err;

            std::cout << "x=" << xi
                      << "  y=" << y[i]
                      << "  y_hat=" << y_hat
                      << "  err=" << err << "\n";
        }
        const double mse = sse / static_cast<double>(m);
        std::cout << "\nSSE = " << sse << "\n";
        std::cout << "MSE = " << mse << "\n";

        // -----------------------------
        // 6) Generar un CSV para graficar
        //
        // fit.csv contendrá:
        //   - puntos originales (x, y)
        //   - curva ajustada en una malla fina (x_fit, y_fit)
        //
        // Esto te permite graficar fácil en Octave:
        //   data = csvread("fit.csv", 1, 0);
		//   x_pts = data(:,1);
        //   y_pts = data(:,2);
        //   x_fit = data(:,3);
        //   y_fit = data(:,4);
        //   plot(x_pts, y_pts, "o"); hold on;
        //   plot(x_fit, y_fit, "-");
        //   grid on;
        //   xlabel("x");
        //   ylabel("y");
        //   title("Ajuste cuadratico");
        //   legend("Datos", "Ajuste");
        // -----------------------------
        {
            std::ofstream out("fit.csv");
            if (!out) throw std::runtime_error("No se pudo crear fit.csv");

            // Encabezados (columnas)
            out << "x_pts,y_pts,x_fit,y_fit\n";

            // Malla fina para la curva
            const int steps = 200;
            const double xmin = x.front();
            const double xmax = x.back();

            for (int i = 0; i < steps; ++i) {
                const double t = static_cast<double>(i) / (steps - 1);
                const double xf = xmin + t * (xmax - xmin);
                const double yf = a * xf * xf + b * xf + c;

                // Para las filas donde haya punto real, lo escribimos; si no, dejamos vacío.
                // (Se hace simple: escribimos puntos reales solo en las primeras x.size() filas)
                if (i < static_cast<int>(x.size())) {
                    out << x[static_cast<size_t>(i)] << "," << y[static_cast<size_t>(i)] << ",";
                } else {
                    out << ",,";
                }
                out << xf << "," << yf << "\n";
            }
        }

        std::cout << "\nSe genero fit.csv para graficar (puntos y curva).\n";

        // Tip rápido de Octave en consola:
		std::cout << "Octave (ejemplo):\n";
		std::cout << "  data = csvread(\"fit.csv\", 1, 0);\n";
		std::cout << "  x_pts = data(:,1);\n";
        std::cout << "  y_pts = data(:,2);\n";
        std::cout << "  x_fit = data(:,3);\n";
        std::cout << "  y_fit = data(:,4);\n";
        std::cout << "  plot(x_pts, y_pts, \"o\"); hold on;\n";
        std::cout << "  plot(x_fit, y_fit, \"-\");\n";
        std::cout << "  grid on;\n";
        std::cout << "  xlabel(\"x\");\n";
        std::cout << "  ylabel(\"y\");\n";
        std::cout << "  title(\"Ajuste cuadratico\");\n";
        std::cout << "  legend(\"Datos\", \"Ajuste\");\n";


        return 0;

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
