#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <png.h>
#include <pthread.h>

#define MAX_POINTS 1000000 
#define MAX_CLASSES 5
#define WIDTH 720
#define HEIGHT 720
#define SIDE 3
#define NUM_THREADS 8
#define K 50

//gcc -o pthreads pthreads.c -lpng -lm -pthread -O3

typedef struct {
    double x, y;
    int class;
} Point;

typedef struct {
    double distance;
    int class;
} DistanceClass;

//struttura dati per distanze
typedef struct {
    Point *points;
    int num_points;
    Point center;
    DistanceClass *distances;
    int start_index;
    int end_index;
} DistanceThreadData;

//struttura dati per classificazione
typedef struct {
    Point *points;
    int num_points;
    int **boundaries;
    int start_row;
    int end_row;
    int k;
} ClassificationThreadData;

double euclidean_distance(Point a, Point b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

void k_bubble_sort(DistanceClass *distances, int size, int k) {
    for (int i = 0; i < k; i++) {
        for (int j = i+1 ; j < size; j++) {
            if (distances[i].distance > distances[j].distance) {
                DistanceClass temp = distances[i];
                distances[i] = distances[j];
                distances[j] = temp;
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
    while (fscanf(file, "%lf,%lf,%d", &temp_points[i].x, &temp_points[i].y, &temp_points[i].class) == 3) {
        i++;
        if (i >= MAX_POINTS) {
            break;
        }
    }

    fclose(file);
    *points = temp_points;
    return i; // Return the number of points read
}

//funzione per calcolare le distanze in parallelo
void *calculate_distances(void *arg) {
    DistanceThreadData *data = (DistanceThreadData *)arg;
    Point center = data->center;
    DistanceClass *distances = data->distances;

    for (int i = data->start_index; i < data->end_index; i++) {
        distances[i].distance = euclidean_distance(data->points[i], center);
        distances[i].class = data->points[i].class;
    }

    return NULL;
}

//funzione per classificare un punto in parallelo
int classify(Point *points, int num_points, Point new_point, int k) {
    DistanceClass *distances = (DistanceClass *)malloc(num_points * sizeof(DistanceClass));
    if (distances == NULL) {
        perror("Unable to allocate memory for distances");
        return -1;
    }

    pthread_t threads[NUM_THREADS];
    DistanceThreadData thread_data[NUM_THREADS];
    int points_per_thread = num_points / NUM_THREADS;

    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].points = points;
        thread_data[i].center = new_point;
        thread_data[i].distances = distances;
        thread_data[i].start_index = i * points_per_thread;
        thread_data[i].end_index = (i == NUM_THREADS - 1) ? num_points : (i + 1) * points_per_thread;

        pthread_create(&threads[i], NULL, calculate_distances, &thread_data[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    k_bubble_sort(distances, num_points, k);

    int counts[MAX_CLASSES] = {0};
    for (int i = 0; i < k; i++) {
        counts[distances[i].class]++;
    }

    int max_count = 0;
    int max_label = -1;
    for (int i = 0; i < MAX_CLASSES; i++) {
        if (counts[i] > max_count) {
            max_count = counts[i];
            max_label = i;
        }
    }

    free(distances);
    return max_label;
}

void *process_rows(void *arg) {
    ClassificationThreadData *data = (ClassificationThreadData *)arg;
    Point *points = data->points;
    int num_points = data->num_points;
    int **boundaries = data->boundaries;
    int start_row = data->start_row;
    int end_row = data->end_row;
    int k = data->k;

    for (int i = start_row; i < end_row; i += SIDE) {
        for (int j = 0; j < WIDTH; j += SIDE) {
            Point center = {i , j }; //pixel di riferimento del quadratino
            if (center.x >= HEIGHT || center.y >= WIDTH) continue;

            int class = classify(points, num_points, center, k); //classifica il punto

            for (int x = i; x < i + SIDE && x < HEIGHT; x++) {
                for (int y = j; y < j + SIDE && y < WIDTH; y++) {
                    boundaries[x][y] = class;
                }
            }
        }
    }

    return NULL;
}

int **get_boundaries(Point *points, int num_points, int h, int w) {
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
    ClassificationThreadData thread_data[NUM_THREADS];
    int rows_per_thread = h / NUM_THREADS;

    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].points = points;
        thread_data[i].num_points = num_points;
        thread_data[i].boundaries = boundaries;
        thread_data[i].start_row = i * rows_per_thread;
        //se è l'ultimo thread prende tutte le righe rimanenti
        thread_data[i].end_row = (i == NUM_THREADS - 1) ? h : (i + 1) * rows_per_thread;
        thread_data[i].k = K;

        pthread_create(&threads[i], NULL, process_rows, &thread_data[i]);
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
            int class = boundaries[x][y];
            if (class == -1) {
                row[x * 4] = 255;
                row[x * 4 + 1] = 255;
                row[x * 4 + 2] = 255;
            } else {
                row[x * 4] = colors[class][0];
                row[x * 4 + 1] = colors[class][1];
                row[x * 4 + 2] = colors[class][2];
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
    const char* filename = "dataset/cinquantamila.csv";

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
