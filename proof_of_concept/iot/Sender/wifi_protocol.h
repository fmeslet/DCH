#ifndef WIFI_PROTOCOL_H
#define WIFI_PROTOCOL_H
#include <Arduino.h>

class WifiProtocol 
{
  private:

    // Define fields
    uint8_t _frameControl[2];
    uint8_t _duration[2];
    uint8_t _dstAddress[6];
    uint8_t _srcAddress[6];
    uint8_t _filtAddress[6]; // Filtering adress (BSSID)
    uint8_t _sequence[2];
    uint8_t _qos[2];
    uint8_t _llc[8];
    
    uint8_t _header[34]; // Set MaxSize
    int _sizeHeader = 0;
    
    uint8_t _payload[300]; // Set MaxSize
    int _sizePayload = 0; 

  public:
    // Constructor
    WifiProtocol();

    // Printer
    void printData();
    void printHeader();
    void printPayload();

    // Getter
    // // Getter array
    uint8_t* getData();
    int getSizeData();
    uint8_t* getHeader();
    int getSizeHeader();
    uint8_t* getPayload();
    int getSizePayload();

    // // Getter others
    uint8_t* getFrameControl();
    uint8_t* getDuration();
    uint8_t* getDstAddress();
    // ...


    // Setter
    void setFrameControl(uint8_t &frameControl);
    void setDuration(uint8_t &duration);
    void setDstAddress(uint8_t &dstAddress); // Do MEMCPY and *
    // ...
    void setHeader(uint8_t *header, 
                      int sizeHeader);
    void setPayload(uint8_t *payload,
                       int sizePayload);
    
};

#endif
