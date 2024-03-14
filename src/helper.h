
#include <iostream>
#include <list>
#include <vector>
#include <map>
#include <type_traits>

#define show(x) std::cout << #x << ": "; print(x)
#define showArr(x, size) std::cout << #x << ": "; print(x, size)



#include <iostream>
#include <type_traits>

template <typename T, typename = void>
struct is_static_array : std::false_type {};

template <typename T>
struct is_static_array<T, std::enable_if_t<std::is_array_v<T> && std::extent_v<T> != 0>> : std::true_type {};


template <typename T>
std::enable_if_t<!is_static_array<T>::value>
print(const T& value) {
    std::cout << value << std::endl;
}

template <typename T, size_t N>
void print(const T (&arr)[N]) {
    printf("hi\n");
    for (size_t i = 0; i < N; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}



template<typename T>
void print(const std::list<T>& itemList, int size=-1) {
    int num = 0;
    for (const auto& item : itemList) {
        if (size > 0 && ++num > size) break;
        std::cout << item << ", ";
    }
    std::cout << std::endl;
}

template<typename T>
void print(const std::vector<T>& itemList, int size=-1) {
    int num = 0;
    for (const auto& item : itemList) {
        if (size > 0 && ++num > size) break;
        std::cout << item << " ";
    }
    std::cout << std::endl;
}

template<typename K, typename V>
void print(const std::map<K, V>& itemMap) {
    for (const auto& item : itemMap) {
        std::cout << item.first << ": " << item.second << std::endl;
    }
}




template <typename T>
double calculateMSE(const T* array1,
                    const T* array2,
                    const size_t size) {

  double mse = 0.0;
  for (size_t i = 0; i < size; i++) {
    const T error = array1[i] - array2[i];
    mse += error * error;
  }
  mse /= size;
  return mse;
}