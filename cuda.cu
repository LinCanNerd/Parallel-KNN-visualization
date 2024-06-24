#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <png.h>
#include <cuda.h>

// Define the maximum number of points and labels
#define MAX_POINTS 1000000 
#define MAX_LABELS 5
#define WIDTH 720
#define HEIGHT 720
#define SIDE 3

// Define the structure of a point
typedef struct {
    double x, y;
    int label;
} Point;

typedef struct {
    double distance;
    int label;
} DistanceLabel;

__device__ double euclidean_distance(Point a, Point b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

__global__ void classify_kernel(Point* d_points, int num_points, int* d_boundaries, int h, int w, int k) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = ty * w + tx;

    if (tx < w && ty < h) {
        Point center = { tx + 0.5, ty + 0.5 };

        DistanceLabel distances[MAX_POINTS];

        for (int i = 0; i < num_points; i++) {
            distances[i].distance = euclidean_distance(d_points[i], center);
            distances[i].label = d_points[i].label;
        }

        // Simple selection sort to find k nearest neighbors
        for (int i = 0; i < k; i++) {
            for (int j = i + 1; j < num_points; j++) {
                if (distances[i].distance > distances[j].distance) {
                    DistanceLabel temp = distances[i];
                    distances[i] = distances[j];
                    distances[j] = temp;
                }
            }
        }

        int counts[MAX_LABELS] = { 0 };
        for (int i = 0; i < k; i++) {
            counts[distances[i].label]++;
        }

        int max_count = 0;
        int max_label = -1;
        for (int i = 0; i < MAX_LABELS; i++) {
            if (counts[i] > max_count) {
                max_count = counts[i];
                max_label = i;
            }
        }

        d_boundaries[tid] = max_label;
    }
}

int compare(const void *a, const void *b) {
    DistanceLabel *da = (DistanceLabel *)a;
    DistanceLabel *db = (DistanceLabel *)b;
    if (da->distance < db->distance) return -1;
    if (da->distance > db->distance) return 1;
    return 0;
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

void draw_boundaries(int* boundaries, png_bytep* row_pointers) {
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
            int label = boundaries[y * WIDTH + x];
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

void write_png_file(const char *file_name, png_bytep* row_pointers) {
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

void free_memory(int* boundaries, png_bytep* row_pointers, Point* points) {
    for (int y = 0; y < HEIGHT; y++) {
        free(row_pointers[y]);
    }
    free(row_pointers);

    free(boundaries);
    free(points);
}

int main() {
    Point* points;
    int num_points;
    const char* filename = "dataset/diecimila.csv";

    num_points = read_csv(filename, &points);
    if (num_points == -1) return 1;

    int* boundaries = (int*)malloc(WIDTH * HEIGHT * sizeof(int));
    if (boundaries == NULL) {
        perror("Unable to allocate memory for boundaries");
        free(points);
        return 1;
    }

    Point* d_points;
    int* d_boundaries;

    cudaMalloc(&d_points, num_points * sizeof(Point));
    cudaMalloc(&d_boundaries, WIDTH * HEIGHT * sizeof(int));

    cudaMemcpy(d_points, points, num_points * sizeof(Point), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x, (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);
    classify_kernel<<<numBlocks, threadsPerBlock>>>(d_points, num_points, d_boundaries, HEIGHT, WIDTH, 5);

    cudaMemcpy(boundaries, d_boundaries, WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);

    png_bytep* row_pointers;
    allocate_memory_for_rows(&row_pointers);

    draw_boundaries(boundaries, row_pointers);
    write_png_file("output/boundariesC.png", row_pointers);

    free_memory(boundaries, row_pointers, points);
    cudaFree(d_points);
    cudaFree(d_boundaries);

    printf("PNG file created successfully!\n");
    return 0;
}
