#ifndef NDEBUG
#define USE_OMP false
#else
#define USE_OMP true
#endif

#define DS_USE_COMPRESSION 1

#include <iostream>
#include <memory>
#include <map>
#include <sstream>
#include <glog/logging.h>
#include <string>

#include <boost/property_tree/ptree.hpp>
#include <boost/algorithm/string.hpp>
#include <restbed>
#include <restbed>
#include <fstream>
#include <map>
#include <array>
#include <vector>
#include <omp.h>

#include "json.hpp"
#include "miniz_wrapper.h"
#include "re2/re2.h"
#include "re2/stringpiece.h"

using std::string;
using std::vector;
using std::unique_ptr;
using std::make_unique;
using std::array;
using json = nlohmann::json;

constexpr size_t kCompressionSizeThreshold = 1400;

std::map<std::string, std::string> file_content;

long MicroSecondsSinceEpoch() {
  return std::chrono::duration_cast<std::chrono::microseconds>
      (std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

const string &ReadCachedFile(const string &filename) {
  auto it = file_content.find(filename);
  if (it == file_content.end()) {
    // Cache miss.
    LOG(INFO) << "Reading " << filename;
    std::ifstream infile(filename);
    std::stringstream buffer;
    buffer << infile.rdbuf();
    file_content[filename] = buffer.str();
    // Strip tags
    RE2::GlobalReplace(&file_content[filename], R"((?i)(?:<span[^>]*?>|</\s*?span>))", "");
    // Reference to static variable.
    return file_content[filename];
  }
  return it->second;
}

bool IsVowel(char c) {
  c = static_cast<char>(std::tolower(c));
  return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
}

void PluralRegex(const string &p, std::ostream &pattern_stream) {
  // Given a singular noun phrase, try to also match its plural form.
  pattern_stream << "(?:";
  if (p.length() > 5) {
    string p1 = p;
    p1.pop_back();
    string p2 = p1;
    p2.pop_back();
    if (p.back() == 's') {
      if (p1.back() == 'i') {
        pattern_stream << p2 << "(?:is|es)";  // -is or -es
      } else {
        pattern_stream << p << "?";  // optional trailing s.
      }
    } else if (p.back() == 'y') {
      if (IsVowel(p1.back())) {
        pattern_stream << p << "s?";  // way, ways
      } else {
        pattern_stream << p1 << "(?:y|ies)";  // y or ies.
      }
    } else if (p.back() == 'o') {
      pattern_stream << p << "(?:s|es)?";  // o can be oes or os.
    } else if (p.back() == 'f') {
      pattern_stream << p1 << "(?:f|ves)";  // f or ves.
    } else if (p.back() == 'x') {
      if (p1.back() == 'i') {
        pattern_stream << p2 << "(?:ices|ix)";  // matrix, matrices
      } else if (p1.back() == 'e') {
        pattern_stream << p2 << "(?:ices|ex)";  // vertex, vertices
      } else {
        pattern_stream << p << "(?:es)?";  // box, boxes
      }
    } else if (p.back() == 'n') {
      if (p1.back() == 'o') {
        // Not always true (e.g. -tion), but this is probably enough.
        pattern_stream << p2 << "(?:on|ons|a)";  // automaton, automata
      } else {
        pattern_stream << p << "(?:s|es)?";  // optional s or es.
      }
    } else {
      pattern_stream << p << "(?:s|es)?";  // optional s or es.
    }
  } else {
    pattern_stream << p << "(?:s|es)?";  // optional s or es.
  }
  pattern_stream << ")";
}

void RegexSearchHandler(const std::shared_ptr<restbed::Session> session) {
  const auto request = session->get_request();
  int content_length = request->get_header("Content-Length", 0);

  session->fetch(
      static_cast<const size_t>(content_length),
      [](const std::shared_ptr<restbed::Session> session, const restbed::Bytes &body) {
        const auto request = session->get_request();
        LOG(INFO) << "POST " << request->get_path();

        const string filename = request->get_path_parameter("filename");
        string f = "/home/daeyun/Dropbox/ipython-notebooks/cs412/project/np_text/" + filename + ".pdf.html";
        const string &s = ReadCachedFile(f);

        if (s.empty()) {
          LOG(WARNING) << "Bad filename";
          session->close(restbed::BAD_REQUEST, "", {{"Content-Length", "0"}});
          return;
        }

        std::string user_str((char *) body.data(), body.size());
        DLOG(INFO) << "Request body: " << user_str;

        std::vector<std::string> patterns;
        try {
          auto parsed = json::parse(user_str);
          patterns.reserve(parsed.size());
          for (const string item : parsed) {
            patterns.push_back(item);
          }

        } catch (const std::exception &exception) {
          LOG(WARNING) << "Exception: " << exception.what();
          session->close(restbed::BAD_REQUEST, "", {{"Content-Length", "0"}});
          return;
        }

        auto __StartTime = MicroSecondsSinceEpoch();
        // Populate local lists.
        vector<unique_ptr<vector<unique_ptr<vector<array<int, 2>>>>>> all_local_ranges;
        for (int i = 0; i < std::min(patterns.size(), static_cast<size_t>(omp_get_max_threads())); ++i) {
          all_local_ranges.push_back(make_unique<vector<unique_ptr<vector<array<int, 2>>>>>());
        }

#pragma omp parallel if (USE_OMP)
        {
          auto *local_ranges = all_local_ranges[omp_get_thread_num()].get();

#pragma omp for
          for (int i = 0; i < patterns.size(); ++i) {
            auto ranges = make_unique<vector<array<int, 2>>>();
            std::stringstream pattern_stream;

            // patterns[i] can be comma-separated.
            pattern_stream << "(?i)\\b(";
            vector<string> strs;
            boost::split(strs, patterns[i], boost::is_any_of(","));
            for (int k = 0; k < strs.size(); ++k) {
              PluralRegex(strs[k], pattern_stream);
              if (k + 1 < strs.size()) {
                pattern_stream << "|";
              }
            }
            pattern_stream << ")\\b";

            RE2 pattern(pattern_stream.str());
            DLOG(INFO) << pattern.pattern();
            re2::StringPiece input(s.data(), static_cast<int>(s.size()));
            const auto *input_ptr = input.data();

            string var;
            while (RE2::FindAndConsume(&input, pattern, &var)) {
              auto end = static_cast<int>(input.data() - input_ptr);
              auto length = static_cast<int>(var.size());
              ranges->push_back({end - length, length});
            }

            local_ranges->push_back(std::move(ranges));
          } // for
        }; // parallel
        LOG(INFO) << "ELAPSED: " << MicroSecondsSinceEpoch() - __StartTime;

        json out_values = json::array();

        for (const auto &local_ranges : all_local_ranges) {
          for (const auto &ranges : *local_ranges) {
            json values = json::array();
            for (const auto &range : *ranges) {
              values.push_back({range[0], range[1]});
            }
            out_values.push_back(values);
          }
        }
        auto response_string = out_values.dump();

# if DS_USE_COMPRESSION
        auto compressed_length = compressBound(response_string.length());
        std::unique_ptr<unsigned char[]> data(new unsigned char[compressed_length]);
        const int status = compress(data.get(),
                                    &compressed_length,
                                    reinterpret_cast<const unsigned char *>(response_string.data()),
                                    static_cast<mz_ulong>(response_string.size()));

        if (status != MZ_OK) {
          LOG(ERROR) << "Compression failed: " << mz_error(status);
          session->close(restbed::BAD_REQUEST, "", {{"Content-Length", "0"}});
          return;
        }

        response_string = std::string(reinterpret_cast<char *>(data.get()), compressed_length);
#endif

        const std::multimap<string, string> headers{
            {"Content-Type", "application/json"},
            {"Content-Length", std::to_string(response_string.length())},
            {"Cache-Control", "no-cache"},
# if DS_USE_COMPRESSION
            {"Content-Encoding", "deflate"},
#endif
        };
        DLOG(INFO) << "OK response. Data: " << response_string;
        session->close(restbed::OK, response_string, headers);
      });
}

void get_handler(const std::shared_ptr<restbed::Session> session) {
  const auto request = session->get_request();
  DLOG(INFO) << "GET " << request->get_path();
  session->close(restbed::OK, "", {{"Connection", "close"}});
}

void option_handler(const std::shared_ptr<restbed::Session> session) {
  const auto request = session->get_request();
  DLOG(INFO) << "OPTION " << request->get_path();
//  auto headers = request->get_headers();
//  for (const auto &kv : headers) {
//    DLOG(INFO) << kv.first;
//    DLOG(INFO) << kv.second;
//    DLOG(INFO) << "";
//  }
  session->close(restbed::OK, "", {{"Connection", "keep-alive"}});
}

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;

  auto resource = std::make_shared<restbed::Resource>();
  resource->set_path("/re2/{filename: [\\w\\-]+}");
  resource->set_method_handler("POST", RegexSearchHandler);

  auto settings = std::make_shared<restbed::Settings>();
  settings->set_port(6600);
  settings->set_default_header("Connection", "close");
  settings->set_default_header("Access-Control-Allow-Origin", "*");
  settings->set_default_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, PATCH, DELETE");
  settings->set_default_header("Access-Control-Allow-Headers", "X-Requested-With,content-type");
  settings->set_default_header("Access-Control-Allow-Credentials", "1");

  resource->set_method_handler("GET", get_handler);
  resource->set_method_handler("OPTIONS", option_handler);

  restbed::Service service;
  service.publish(resource);
  service.start(settings);

  return EXIT_SUCCESS;
}