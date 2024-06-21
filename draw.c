#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <png.h>
#include <math.h>

#define WIDTH 720
#define HEIGHT 720
#define RADIUS 5

typedef struct {
    int x;
    int y;
    int class;
} Point;

// Function to draw a circle in the image
void draw_circle(png_bytep *row_pointers, int cx, int cy, int radius, png_byte r, png_byte g, png_byte b) {
    png_byte edge_r = r / 2;
    png_byte edge_g = g / 2;
    png_byte edge_b = b / 2;

    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            int distance_sq = x * x + y * y;
            if (distance_sq <= radius * radius) {
                int px = cx + x;
                int py = cy + y;
                if (px >= 0 && px < WIDTH && py >= 0 && py < HEIGHT) {
                    png_bytep px_ptr = &(row_pointers[py][px * 4]);

                    // Determine if the current pixel is near the edge
                    double distance = sqrt((double)distance_sq);
                    if (distance > radius * 0.8) {
                        // Darker color for the edge
                        px_ptr[0] = edge_r;
                        px_ptr[1] = edge_g;
                        px_ptr[2] = edge_b;
                    } else {
                        // Lighter color for the inside
                        px_ptr[0] = r;
                        px_ptr[1] = g;
                        px_ptr[2] = b;
                    }
                    px_ptr[3] = 255;
                }
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

    Point *temp_points = (Point *)malloc(10000 * sizeof(Point));
    if (temp_points == NULL) {
        perror("Unable to allocate memory for points");
        fclose(file);
        return -1;
    }

    int i = 0;
    char line[128];
    while (fgets(line, sizeof(line), file)) {
        if (sscanf(line, "%d,%d,%d", &temp_points[i].x, &temp_points[i].y, &temp_points[i].class) == 3) {
            i++;
            if (i >= 10000) {
                break;
            }
        }
    }

    fclose(file);
    *points = temp_points;
    return i; // Return the number of points read
}

int main() {
    Point *points;
    int num_points = read_csv("dataset/diecimila.csv", &points);
    if (num_points < 0) {
        return 1; // Error reading CSV file
    }

    // Colors for each class
    png_byte colors[6][3] = {
        {255, 0, 0},    // Red
        {0, 255, 0},    // Green
        {0, 0, 255},    // Blue
        {255, 255, 0},  // Yellow
        {0, 255, 255},  // Cyan
        {255, 0, 255}   // Magenta
    };

    // Allocate memory for image
    png_bytep row_pointers[HEIGHT];
    for (int y = 0; y < HEIGHT; y++) {
        row_pointers[y] = (png_bytep)malloc(WIDTH * 4 * sizeof(png_byte));
        for (int x = 0; x < WIDTH; x++) {
            png_bytep px = &(row_pointers[y][x * 4]);
            px[0] = 255; // R
            px[1] = 255; // G
            px[2] = 255; // B
            px[3] = 255; // A
        }
    }

    // Draw circles for each point
    for (int i = 0; i < num_points; i++) {
        Point p = points[i];
        if (p.class >= 0 && p.class < 5) {
            png_byte *color = colors[p.class];
            draw_circle(row_pointers, p.x, p.y, RADIUS, color[0], color[1], color[2]);
        }
    }

    // Write image to file
    FILE *fp = fopen("output/output.png", "wb");
    if (!fp) {
        perror("Unable to open output file");
        return 1;
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fclose(fp);
        return 1;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, NULL);
        fclose(fp);
        return 1;
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return 1;
    }

    png_init_io(png, fp);

    // Write header
    png_set_IHDR(
        png,
        info,
        WIDTH, HEIGHT,
        8,
        PNG_COLOR_TYPE_RGBA,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );
    png_write_info(png, info);

    // Write bytes
    png_write_image(png, row_pointers);

    // End write
    png_write_end(png, NULL);

    // Free allocated memory
    for (int y = 0; y < HEIGHT; y++) {
        free(row_pointers[y]);
    }

    fclose(fp);
    png_destroy_write_struct(&png, &info);
    free(points);

    return 0;
}
