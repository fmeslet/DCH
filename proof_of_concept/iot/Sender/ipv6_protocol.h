#ifndef IPv6_PROTOCOL_H
#define IPv6_PROTOCOL_H
#include <Arduino.h>

#define IPv6_PROTO_TCP_V     0x06
#define IPv6_PROTO_UDP_V     0x11
#define IPv6_HEADER_LEN_V    0x28

// Definir toutes les fields positions
#define IPv6_VERSION_TC_P    0x00 // 0 and 
#define IPv6_TC_FL_P         0x01 // 1 
#define IPv6_PAYLOAD_LEN_P   0x04 // 4 
#define IPv6_NEXT_HEADER_P   0x06 // 6 
#define IPv6_HOP_LIMIT_P     0x07 // 7 
#define IPv6_SRC_P           0x08 // 8
#define IPv6_DST_P           0x18 // 24

/*
   For improvement: https://github.com/njh/EtherSia/blob/main/src/IPv6Address.cpp
*/

class Ipv6Protocol 
{
  private:

    // Define fields
    uint8_t _versionTrafficClass;
    uint8_t _trafficClassFlowLabel[3];
    uint8_t _payloadLength[2];
    uint8_t _nextHeader[2];
    uint8_t _hopLimit;
    uint8_t _srcAddress[16];
    uint8_t _dstAddress[16];
    
    uint8_t _header[40]; // Set MaxSize
    int _sizeHeader = 0;
    
    uint8_t _payload[1500]; // Set MaxSize
    int _sizePayload = 0; 

    // Private Methods
    void computeAddressEui64(
      uint8_t* eui64Address, 
      uint8_t* macAddress);
    void computeAddressLocalLink(
      uint8_t* ipv6Address,
      uint8_t* macAddress);

  public:
    // Constructor
    Ipv6Protocol();

    // Others
    void resetPayloadLength();
    uint8_t* convertBinToHex(
         uint8_t* arrayBin,
         int sizeArrayBin);
    uint8_t* convertIntToBin(
           uint16_t value);

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
    uint8_t getVersionTrafficClass();
    uint8_t* getTrafficClassFlowLabel();
    uint8_t* getPayloadLength();
    // ...

    
    // Setter
    void setVersionTrafficClass(
      uint8_t versionTrafficClass);
    void setTrafficClassFlowLabel(
      uint8_t* trafficClassFlowLabel);
    void setPayloadLength(
      uint8_t *payloadLength);
    void setNextHeader(
      uint8_t* nextHeader);
    void setHopLimit(
      uint8_t hopLimit);
    void setSrcAddressLocalLink(
      uint8_t *macAddress);
    void setSrcAddress(
      uint8_t *srcAddress);
    void setDstAddressLocalLink(
      uint8_t *macAddress);
    void setDstAddress(
      uint8_t *dstAddress);
    void setHeader(uint8_t *header);
    void setPayload(uint8_t *payload,
                    int sizePayload,
                    bool resetPayloadLength);
    
};

#endif
