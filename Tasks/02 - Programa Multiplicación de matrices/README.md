# Instalación de MSYS2, MinGW (UCRT64), g++23, OpenBLAS, LAPACK y LAPACKE en Windows 11

Este documento describe cómo **instalar desde cero** y **actualizar** un entorno en **Windows 11**
para desarrollar en **C++23** usando **DGEMM**, **OpenBLAS**, **LAPACK** y **LAPACKE**.

---

## 1. Descargar e instalar MSYS2 (oficial)

### Descarga MSYS2 desde su página oficial:

👉 https://www.msys2.org/

### Pasos
1. Descarga el instalador `msys2-x86_64-*.exe`
2. Instálalo en una ruta simple, por ejemplo:

C:\msys64

3. Al finalizar, **abre desde el menú inicio**:

MSYS2 UCRT64


⚠️ **No uses** la terminal “MSYS” ni “MINGW64”.  
Para este proyecto es obligatorio **UCRT64**.

---

## 2. Actualizar MSYS2

En la terminal **MSYS2 UCRT64**, ejecuta:

```bash
pacman -Syu
```

Si la terminal se cierra automáticamente, vuelve a abrir MSYS2 UCRT64 y ejecuta otra vez:

```bash
pacman -Syu
```

## 3. Instalar MinGW-w64 (UCRT64) y g++ (C++23)

### Instala el toolchain completo para UCRT64:

```bash
pacman -S --needed mingw-w64-ucrt-x86_64-toolchain
```

Esto instala:

- gcc
- g++
- make
- gdb

Verificar g++

```bash
g++ --version
```

Para compilar con C++23, se usa:

-std=c++23

## 4. Instalar OpenBLAS, LAPACK y LAPACKE

### Instala las librerías de álgebra lineal:

```bash
pacman -S --needed \
  mingw-w64-ucrt-x86_64-openblas \
  mingw-w64-ucrt-x86_64-lapack \
  mingw-w64-ucrt-x86_64-lapacke
```

¿Para qué sirve cada una?

OpenBLAS
Implementación optimizada de BLAS (incluye DGEMM).

LAPACK
Rutinas avanzadas de álgebra lineal (usa BLAS internamente).

LAPACKE
Interfaz en C para LAPACK (lapacke.h).

## 5. Verificar instalación

### Ver paquetes instalados

```bash
pacman -Qi mingw-w64-ucrt-x86_64-openblas
pacman -Qi mingw-w64-ucrt-x86_64-lapack
pacman -Qi mingw-w64-ucrt-x86_64-lapacke
```

### Verificar header LAPACKE

```bash
pacman -Ql mingw-w64-ucrt-x86_64-lapacke | grep lapacke.h
```

## 6. Compilar programas con DGEMM (BLAS)

### Usando DGEMM (OpenBLAS)

```bash
g++ -std=c++23 -O2 -Wall -Wextra main.cpp -o app.exe -lopenblas
```

Esto permite usar:
- dgemm_ (BLAS Fortran)
- cblas_dgemm (CBLAS)

## 7. Compilar programas con LAPACKE

Si tu código usa funciones LAPACKE (por ejemplo LAPACKE_dgesv):

```bash
g++ -std=c++23 -O2 -Wall -Wextra main-multiplicacion.cpp -o app.exe -llapacke -lopenblas
```

Si hay problemas de enlace, prueba este orden:

```bash
g++ -std=c++23 -O2 -Wall -Wextra main-multiplicacion.cpp -o app.exe -lopenblas -llapacke
```

## 8. Ejecuta el programa de multiplicación de matrices

### Formato para editar `input.txt`

Para ingresar M1 (mxn) M2 (nxl):

m n l
M1
M2

Ejemplo: 

2 3 2
1 2 3
1 1 1
2 3
3 4
5 6

```bash
./app.exe input.txt output.txt
```
### Formato de salida `ouput.txt`

Buscar el archivo en la carpeta raiz de tu programa:

el formato de salida sería:
m l
M3

Ejemplo:
2 2
12 13
56 20