#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <png.h>
#include <pthread.h>

// Define the maximum number of points and labels
#define MAX_POINTS 1000000 
#define MAX_LABELS 5
#define WIDTH 720
#define HEIGHT 720
#define SIDE 3
#define NUM_THREADS 8

// Define the structure of a point
typedef struct {
    double x, y;
    int label;
} Point;

typedef struct {
    double distance;
    int label;
} DistanceLabel;

typedef struct {
    int thread_id;
    int start_row;
    int end_row;
    Point *points;
    int num_points;
    int **boundaries;
} ThreadData;

int compare(const void *a, const void *b) {
    DistanceLabel *da = (DistanceLabel *)a;
    DistanceLabel *db = (DistanceLabel *)b;
    if (da->distance < db->distance) return -1;
    if (da->distance > db->distance) return 1;
    return 0;
}

void insertion_sort(DistanceLabel *distances, int n) {
    for (int i = 1; i < n; i++) {
        DistanceLabel key = distances[i];
        int j = i - 1;
        while (j >= 0 && distances[j].distance > key.distance) {
            distances[j + 1] = distances[j];
            j = j - 1;
        }
        distances[j + 1] = key;
    }
}

double euclidean_distance(Point a, Point b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
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

int classify(Point *points, int num_points, Point new_point, int k) {
    DistanceLabel *distances = (DistanceLabel *)malloc(num_points * sizeof(DistanceLabel));
    if (distances == NULL) {
        perror("Unable to allocate memory for distances");
        return -1;
    }

    for (int i = 0; i < num_points; i++) {
        distances[i].distance = euclidean_distance(points[i], new_point);
        distances[i].label = points[i].label;
    }

    qsort(distances, num_points, sizeof(DistanceLabel), compare);

    int counts[MAX_LABELS] = {0};
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

    free(distances);
    return max_label;
}

void* classify_subrows(void* arg) {
    ThreadData *data = (ThreadData*)arg;

    for (int i = data->start_row; i < data->end_row; i += SIDE) {
        for (int j = 0; j < WIDTH; j += SIDE) {
            Point center = {i + 1, j + 1}; // Center of 3x3 square
            if (center.x >= HEIGHT || center.y >= WIDTH) continue; // Skip if center is out of bounds
            int class = classify(data->points, data->num_points, center, 5);
            for (int x = i; x < i + SIDE && x < HEIGHT; x++) {
                for (int y = j; y < j + SIDE && y < WIDTH; y++) {
                    data->boundaries[x][y] = class;
                }
            }
        }
    }

    pthread_exit(NULL);
}

int** get_boundaries(Point *points, int num_points, int h, int w) {
    int **boundaries = (int **)malloc(h * sizeof(int *));
    
    if (boundaries == NULL) {
        perror("Unable to allocate memory for boundaries");
        return NULL;
    }
    
    for (int i = 0; i < h; i++) {
        boundaries[i] = (int *)malloc(w * sizeof(int));
        if (boundaries[i] == NULL) {
            perror("Unable to allocate memory for boundaries row");
            for (int j = 0; j < i; j++) {
                free(boundaries[j]);
            }
            free(boundaries);
            return NULL;
        }
    }

    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];
    int rows_per_thread = HEIGHT / NUM_THREADS;

    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].start_row = i * rows_per_thread;
        thread_data[i].end_row = (i + 1) * rows_per_thread;
        thread_data[i].points = points;
        thread_data[i].num_points = num_points;
        thread_data[i].boundaries = boundaries;

        pthread_create(&threads[i], NULL, classify_subrows, (void*)&thread_data[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    return boundaries;
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
            int label = boundaries[x][y];
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

void free_memory(int **boundaries, png_bytep *row_pointers, Point *points) {
    for (int y = 0; y < HEIGHT; y++) {
        free(row_pointers[y]);
    }
    free(row_pointers);

    for (int i = 0; i < HEIGHT; i++) {
        free(boundaries[i]);
    }
    free(boundaries);
    free(points);
}

int main() {
    Point *points;
    int num_points;
    const char* filename = "dataset/diecimila.csv";

    num_points = read_csv(filename, &points);
    if (num_points == -1) return 1;

    int **boundaries = get_boundaries(points, num_points, HEIGHT, WIDTH);
    if (boundaries == NULL) {
        free(points);
        return 1;
    }

    png_bytep *row_pointers;
    allocate_memory_for_rows(&row_pointers);

    draw_boundaries(boundaries, row_pointers);
    write_png_file("output/boundariesP.png", row_pointers);

    free_memory(boundaries, row_pointers, points);

    printf("PNG file created successfully!\n");
    return 0;
}
