#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <png.h>
#include <cuda_runtime.h>

#define MAX_POINTS 1000000
#define MAX_LABELS 5
#define WIDTH 720
#define HEIGHT 720
#define SIDE 4
#define TILE_SIZE 32 // Each tile will be 32x32 pixels

typedef struct {
    double x, y;
    int label;
} Point;

__device__ double euclidean_distance(Point a, Point b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

__device__ void insertion_sort(double *distances, int *labels, int k) {
    for (int i = 1; i < k; i++) {
        double key_dist = distances[i];
        int key_label = labels[i];
        int j = i - 1;
        while (j >= 0 && distances[j] > key_dist) {
            distances[j + 1] = distances[j];
            labels[j + 1] = labels[j];
            j--;
        }
        distances[j + 1] = key_dist;
        labels[j + 1] = key_label;
    }
}

__device__ int classify_gpu(Point *points, int num_points, Point new_point, int k) {
    double distances[MAX_LABELS];
    int labels[MAX_LABELS];

    // Initialize distances and labels with the first k points
    for (int i = 0; i < k; i++) {
        distances[i] = euclidean_distance(points[i], new_point);
        labels[i] = points[i].label;
    }

    // Sort initial k points
    insertion_sort(distances, labels, k);

    // Compare remaining points
    for (int i = k; i < num_points; i++) {
        double dist = euclidean_distance(points[i], new_point);
        if (dist < distances[k - 1]) {
            distances[k - 1] = dist;
            labels[k - 1] = points[i].label;
            insertion_sort(distances, labels, k);
        }
    }

    // Count labels
    int counts[MAX_LABELS] = {0};
    for (int i = 0; i < k; i++) {
        counts[labels[i]]++;
    }

    // Find the most frequent label
    int max_count = 0;
    int max_label = -1;
    for (int i = 0; i < MAX_LABELS; i++) {
        if (counts[i] > max_count) {
            max_count = counts[i];
            max_label = i;
        }
    }

    return max_label;
}

__global__ void get_boundaries_gpu(Point *points, int num_points, int *boundaries, int width, int height) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int x = bx * TILE_SIZE + tx * SIDE;
    int y = by * TILE_SIZE + ty * SIDE;

    if (x < width && y < height) {
        Point center = {(double)x + 1.5, (double)y + 1.5}; // Center of 3x3 square
        int label = classify_gpu(points, num_points, center, 5);

        for (int dx = 0; dx < SIDE && x + dx < width; dx++) {
            for (int dy = 0; dy < SIDE && y + dy < height; dy++) {
                boundaries[(y + dy) * width + (x + dx)] = label;
            }
        }
    }
}

int read_csv(const char *filename, Point **points) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Unable to open file");
        return -1;
    }

    Point *temp_points = (Point *)malloc(MAX_POINTS * sizeof(Point));
    if (temp_points == NULL) {
        perror("Unable to allocate memory for points");
        fclose(file);
        return -1;
    }

    int i = 0;
    while (fscanf(file, "%lf,%lf,%d", &temp_points[i].x, &temp_points[i].y, &temp_points[i].label) == 3) {
        i++;
        if (i >= MAX_POINTS) {
            break;
        }
    }

    fclose(file);
    *points = temp_points;
    return i; // Return the number of points read
}

void draw_boundaries(int **boundaries, png_bytep *row_pointers) {
    png_byte colors[6][3] = {
        {255, 0, 0},    // Red
        {0, 255, 0},    // Green
        {0, 0, 255},    // Blue
        {255, 255, 0},  // Yellow
        {0, 255, 255},  // Cyan
        {255, 0, 255}   // Magenta
    };

    for (int y = 0; y < HEIGHT; y++) {
        png_bytep row = row_pointers[y];
        for (int x = 0; x < WIDTH; x++) {
            int label = boundaries[y][x];
            if (label == -1) {
                row[x * 4] = 255;
                row[x * 4 + 1] = 255;
                row[x * 4 + 2] = 255;
            } else {
                row[x * 4] = colors[label][0];
                row[x * 4 + 1] = colors[label][1];
                row[x * 4 + 2] = colors[label][2];
            }
            row[x * 4 + 3] = 255;
        }
    }
}

void write_png_file(const char *file_name, png_bytep *row_pointers) {
    FILE *fp = fopen(file_name, "wb");
    if (!fp) {
        perror("Unable to open file for writing");
        exit(EXIT_FAILURE);
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        perror("Unable to create PNG write structure");
        exit(EXIT_FAILURE);
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        perror("Unable to create PNG info structure");
        png_destroy_write_struct(&png, NULL);
        exit(EXIT_FAILURE);
    }

    if (setjmp(png_jmpbuf(png))) {
        perror("Error during PNG creation");
        png_destroy_write_struct(&png, &info);
        exit(EXIT_FAILURE);
    }

    png_init_io(png, fp);
    png_set_IHDR(png, info, WIDTH, HEIGHT, 8, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    for (int y = 0; y < HEIGHT; y++) {
        png_write_row(png, row_pointers[y]);
    }

    png_write_end(png, NULL);
    fclose(fp);
    png_destroy_write_struct(&png, &info);
}

void allocate_memory_for_rows(png_bytep **row_pointers) {
    *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * HEIGHT);
    if (*row_pointers == NULL) {
        perror("Unable to allocate memory for row pointers");
        exit(EXIT_FAILURE);
    }
    
    for (int y = 0; y < HEIGHT; y++) {
        (*row_pointers)[y] = (png_byte *)malloc(WIDTH * 4 * sizeof(png_byte));
        if ((*row_pointers)[y] == NULL) {
            perror("Unable to allocate memory for PNG rows");
            for (int j = 0; j < y; j++) {
                free((*row_pointers)[j]);
            }
            free(*row_pointers);
            exit(EXIT_FAILURE);
        }
    }
}

int main() {
    Point *h_points;
    int num_points;
    const char* filename = "dataset/cinquantamila.csv";

    num_points = read_csv(filename, &h_points);
    if (num_points == -1) return 1;

    // Allocate device memory
    Point *d_points;
    cudaMalloc(&d_points, num_points * sizeof(Point));
    cudaMemcpy(d_points, h_points, num_points * sizeof(Point), cudaMemcpyHostToDevice);

    int *d_boundaries;
    cudaMalloc(&d_boundaries, WIDTH * HEIGHT * sizeof(int));

    // Launch kernel
    dim3 block_size(TILE_SIZE / SIDE, TILE_SIZE / SIDE);
    dim3 num_blocks((WIDTH + TILE_SIZE) / TILE_SIZE, (HEIGHT + TILE_SIZE) / TILE_SIZE);
    get_boundaries_gpu<<<num_blocks, block_size>>>(d_points, num_points, d_boundaries, WIDTH, HEIGHT);

    // Check for any errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Ensure the kernel has completed
    cudaDeviceSynchronize();

    // Copy result back to host
    int *h_boundaries = (int *)malloc(WIDTH * HEIGHT * sizeof(int));
    cudaMemcpy(h_boundaries, d_boundaries, WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);

    // Convert 1D array to 2D for draw_boundaries function
    int **boundaries_2d = (int **)malloc(HEIGHT * sizeof(int *));
    for (int i = 0; i < HEIGHT; i++) {
        boundaries_2d[i] = &h_boundaries[i * WIDTH];
    }

    png_bytep *row_pointers;
    allocate_memory_for_rows(&row_pointers);

    draw_boundaries(boundaries_2d, row_pointers);
    write_png_file("output/boundariesC.png", row_pointers);

    // Free memory
    cudaFree(d_points);
    cudaFree(d_boundaries);
    free(h_points);
    free(h_boundaries);
    free(boundaries_2d);
    for (int y = 0; y < HEIGHT; y++) {
        free(row_pointers[y]);
    }
    free(row_pointers);

    printf("PNG file created successfully using CUDA!\n");
    return 0;
}