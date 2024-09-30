#include <fstream>
#include <vector>
#include <string>
#include <ctime>
#include <iomanip>
#include <thread>
#include <mutex>
#include <sstream>

#include "file_utils.h"
#include "../constants.h"

size_t estimated_lines = 1500000; // Pre-allocate space for efficiency

// Faster strptime equivalent: manually parsing the string
bool fastStrptime(const char* s, struct tm* tm) {
    // Initialize the tm structure to zero
    memset(tm, 0, sizeof(struct tm));

    // Example format: "YYYY-MM-DD HH:MM:SS"
    int year, month, day, hour, minute, second;

    // Parse the string using sscanf (or manually split and convert the values)
    if (sscanf(s, "%d-%d-%d %d:%d:%d", &year, &month, &day, &hour, &minute, &second) != 6) {
        return false;
    }

    // Set the tm structure fields
    tm->tm_year = year - 1900; // Years since 1900
    tm->tm_mon  = month - 1;   // Months are 0-based
    tm->tm_mday = day;
    tm->tm_hour = hour;
    tm->tm_min  = minute;
    tm->tm_sec  = second;

    return true;
}

// Function to convert date string to a time_t value (elapsed seconds since epoch)
tm stringToTime(const std::string& dateTimeStr) {
    tm tm;
    fastStrptime(dateTimeStr.c_str(), &tm);
    return tm;
}

// Mutexes to protect shared vectors and line counts
std::mutex y_mutex;

void processChunk(const std::vector<std::string>& lines,
                  std::vector<std::vector<double>>& X,
                  std::vector<double>& y,
                  int thread_id,
                  int total_lines) {
    std::vector<std::vector<double>> local_X;
    local_X.reserve(total_lines);
    std::vector<double> local_y;
    local_y.reserve(total_lines);

    for (const auto& line : lines) {
        std::stringstream ss(line);
        std::string token;
        std::vector<double> row;

        std::getline(ss, token, ','); // Skip id
        std::getline(ss, token, ','); // Skip vendor_id
        std::string pickup_datetime, dropoff_datetime;
        std::getline(ss, pickup_datetime, ','); // pickup_datetime
        std::getline(ss, dropoff_datetime, ','); // dropoff_datetime

        tm pickup_time = stringToTime(pickup_datetime);
        row.push_back(pickup_time.tm_mon);
        row.push_back(pickup_time.tm_mday);
        row.push_back(pickup_time.tm_hour);
        row.push_back(pickup_time.tm_min);

        tm dropoff_time = stringToTime(dropoff_datetime);
        row.push_back(dropoff_time.tm_mon);
        row.push_back(dropoff_time.tm_mday);
        row.push_back(dropoff_time.tm_hour);
        row.push_back(dropoff_time.tm_min);

        std::getline(ss, token, ','); // passenger_count
        row.push_back(std::stod(token));
        std::getline(ss, token, ','); // pickup_longitude
        row.push_back(std::stod(token));
        std::getline(ss, token, ','); // pickup_latitude
        row.push_back(std::stod(token));
        std::getline(ss, token, ','); // dropoff_longitude
        row.push_back(std::stod(token));
        std::getline(ss, token, ','); // dropoff_latitude
        row.push_back(std::stod(token));

        std::getline(ss, token, ','); // store_and_fwd_flag (skip this)

        // Finally, the dependent variable: trip_duration
        std::getline(ss, token, ',');
        local_y.push_back(std::stod(token));
        local_X.push_back(row);
    }

    std::lock_guard<std::mutex> lock(y_mutex);
    y.insert(y.end(), local_y.begin(), local_y.end());
    X.insert(X.end(), local_X.begin(), local_X.end());
}

std::vector<std::string> readLines(const std::string& filename) {
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file.");
    }

    const size_t buffer_size = 10 * 1024 * 1024; // 10MB buffer size for reading chunks of the file
    std::vector<char> buffer(buffer_size);
    std::string leftover;

    std::vector<std::string> lines;
    size_t total_bytes_read = 0;

    bool header_skipped = false;
    while (file.read(buffer.data(), buffer.size()) || file.gcount() > 0) {
        size_t bytes_read = file.gcount();
        total_bytes_read += bytes_read;

        // Convert buffer to string, append any leftover from previous read
        std::string chunk = leftover + std::string(buffer.data(), bytes_read);

        // Split the chunk into lines
        size_t start = 0, end = 0;
        while ((end = chunk.find('\n', start)) != std::string::npos) {
            if (!header_skipped) {
                // Skip the first line (header)
                header_skipped = true;
            } else {
                lines.push_back(chunk.substr(start, end - start));
            }

            start = end + 1;
        }

        // Any leftover after the last newline (incomplete line), save for next chunk
        leftover = chunk.substr(start);
    }

    // If there's any leftover at the end of the file, add it as a final line
    if (!leftover.empty()) {
        lines.push_back(leftover);
    }

    return lines;
}

std::vector<std::vector<double>> readCSV(const std::string& filename, std::vector<double>& y) {
    std::vector<std::string> lines = readLines(filename);

    int num_threads = FILE_READ_THREAD_COUNT; // Use the number of hardware threads available
    std::vector<std::thread> threads;
    std::vector<std::vector<double>> X;

    y.reserve(estimated_lines);
    X.reserve(estimated_lines);

    // Split the lines into chunks for each thread
    size_t lines_per_thread = lines.size() / num_threads;
    size_t remainder = lines.size() % num_threads;

    size_t start_index = 0;
    for (int i = 0; i < num_threads; ++i) {
        size_t end_index = start_index + lines_per_thread + (i < remainder ? 1 : 0);
        std::vector<std::string> thread_lines(lines.begin() + start_index, lines.begin() + end_index);

        // Launch thread to process its chunk
        threads.emplace_back(processChunk, thread_lines, std::ref(X), std::ref(y), i + 1, end_index - start_index);

        start_index = end_index;
    }

    // Join all threads
    for (auto& thread : threads) {
        thread.join();
    }

    return X;
}