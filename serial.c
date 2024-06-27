#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <png.h>

//gcc -o serial serial.c -lpng -lm

// Definisco il numero massimo di punti e class
#define MAX_POINTS 1000000 
#define MAX_CLASSES 5
#define WIDTH 720
#define HEIGHT 720
#define SIDE 3
#define K 5

// Definisco la struttra di un punto
typedef struct {
    double x, y;
    int class;
} Point;

// Definisco la struttura di una distanza con etichetta
typedef struct {
    double distance;
    int class;
} DistanceLabel;

//bubble sort limitato a K iterazioni
void k_bubble_sort(DistanceLabel *distances, int size, int k) {
    for (int i = 0; i < k; i++) {
        for (int j = i+1 ; j < size; j++) {
            if (distances[i].distance > distances[j].distance) {
                DistanceLabel temp = distances[i];
                distances[i] = distances[j];
                distances[j] = temp;
            }
        }
    }
}

//calcolo della distanza euclidea
double euclidean_distance(Point a, Point b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

//funzione per leggere il file csv
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

//funzione per classificare un punto
int classify(Point *points, int num_points, Point new_point, int k) {
    DistanceLabel *distances = (DistanceLabel *)malloc(num_points * sizeof(DistanceLabel));
    if (distances == NULL) {
        perror("Unable to allocate memory for distances");
        return -1;
    }

    //calcolo delle distanze euclidiane
    for (int i = 0; i < num_points; i++) {
        distances[i].distance = euclidean_distance(points[i], new_point);
        distances[i].class = points[i].class;
    }

    k_bubble_sort(distances, num_points, k);


    //conta le classi e prendi quello maggiore
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


//funzione per calcolare i confini
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

    for (int i = 0; i < h; i += SIDE) {
        for (int j = 0; j < w; j += SIDE) {
            Point center = {i, j}; // Punto di riferimento del quadratino
            if (center.x >= h || center.y >= w) continue; // controllo i confini
            int class = classify(points, num_points, center, K);
            for (int x = i; x < i + SIDE && x < h; x++) {
                for (int y = j; y < j + SIDE && y < w; y++) {
                    boundaries[x][y] = class;
                }
            }
        }
    }

    return boundaries;
}


//colora i confini
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

//scrive il file png
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
    //leggo il file csv
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
    write_png_file("output/boundaries.png", row_pointers);

    free_memory(boundaries, row_pointers, points);

    printf("PNG file created successfully!\n");
    return 0;
}