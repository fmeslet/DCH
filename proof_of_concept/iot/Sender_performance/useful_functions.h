#ifndef USEFUL_FUNCTIONS_H
#define USEFUL_FUNCTIONS_H
#include <Arduino.h>

void printArray(uint8_t* myArray, int sizeArray);
void printArray(char* myArray, int sizeArray);
uint8_t* ipAddressStringToHex(String ipAddressString, 
                              int sizeIpAddressArray,
                              char separator);
uint8_t* convertHexaToBit(uint8_t* arrayHex,
                          int sizeArrayHex);

#endif
