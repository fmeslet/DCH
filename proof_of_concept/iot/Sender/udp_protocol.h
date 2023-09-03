#ifndef UDP_PROTOCOL_H
#define UDP_PROTOCOL_H
#include <Arduino.h>

#define UDP_PROTO_TCP_V     0x06
#define UDP_PROTO_UDP_V     0x11
#define UDP_HEADER_LEN_V    0x08

// Definir toutes les fields positions
#define UDP_SRC_PORT_P      0x00 // 12
#define UDP_DST_PORT_P      0x02 // 16
#define UDP_LENGTH_P        0x04 // 1 
#define UDP_CHECKSUM_P      0x06 // 10

class UdpProtocol 
{
  private:

    // Define fields
    uint8_t _srcPort[2];
    uint8_t _dstPort[2];
    uint8_t _length[2];
    uint8_t _checksum[2];
    
    uint8_t _header[8]; // Set MaxSize
    
    uint8_t _payload[100]; // Set MaxSize
    int _sizePayload = 0; 

    // Private Methods
    uint16_t computeChecksum(
      uint8_t* buf, uint16_t len, uint8_t type);
    uint8_t* convertBinToHex(
      uint8_t* arrayBin, int sizeArrayBin);
    uint8_t* convertIntToBin(uint16_t value);

  public:
    // Constructor
    UdpProtocol();

    // Others
    void resetChecksum();
    void resetLength();

    // Printer
    void printData();
    void printHeader();
    void printPayload();

    // Getter
    // // Getter array
    uint8_t* getData();
    int getSizeData();
    uint8_t* getHeader();
    uint8_t* getPayload();
    int getSizePayload();

    // // Getter others
    uint8_t* getSrcPort();
    uint8_t* getDstPort();
    uint8_t* getLength();
    uint8_t* getChecksum();

    // Setter
    void setSrcPort(uint8_t *srcPort,
                    bool resetChecksum);
    void setDstPort(uint8_t *dstPort,
                    bool resetChecksum);
    void setLength(uint8_t *length,
                   bool resetChecksum);
    void setChecksum(uint8_t *checksum,
                     bool resetChecksum);

    void setHeader(uint8_t *header, 
                   bool resetChecksum);
    void setPayload(uint8_t *payload,
                    int sizePayload,
                    bool resetChecksum,
                    bool resetLength);
    
};

#endif
