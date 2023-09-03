#ifndef IPv4_PROTOCOL_H
#define IPv4_PROTOCOL_H
#include <Arduino.h>

#define IPv4_PROTO_TCP_V     0x06
#define IPv4_PROTO_UDP_V     0x11
#define IPv4_HEADER_LEN_V    0x14

// Definir toutes les fields positions
#define IPv4_VERSION_P       0x00 // 0 version and header length
#define IPv4_SERVICE_P       0x02 // 1 
#define IPv4_LENGTH_P        0x02 // 2 header length + payload length
#define IPv4_ID_P            0x04 // 4 
#define IPv4_FLAGS_P         0x06 // 6 
#define IPv4_FRAGMENT_P      0x07 // 7 
#define IPv4_TTL_P           0x08 // 8 
#define IPv4_PROTOCOL_P      0x09 // 9 
#define IPv4_CHECKSUM_P      0x0a // 10
#define IPv4_SRC_P           0x0c // 12
#define IPv4_DST_P           0x0f // 16

class Ipv4Protocol 
{
  private:

    // Define fields
    uint8_t _versionProto;
    uint8_t _serviceField;
    uint8_t _totalLength[2];
    uint8_t _identification[2];
    uint8_t _flags;
    uint8_t _fragmentOffset;
    uint8_t _ttl;
    uint8_t _protocol;
    uint8_t _checksum[2];
    uint8_t _srcAddress[4];
    uint8_t _dstAddress[4];
    
    uint8_t _header[20]; // Set MaxSize
    int _sizeHeader = 0;
    
    uint8_t _payload[1500]; // Set MaxSize
    int _sizePayload = 0; 

    // Private Methods
    uint16_t computeChecksum(
      uint8_t* buf, uint16_t len, uint8_t type);
    uint8_t* ipAddressStringToHex(
      String ipAddressString, 
      int sizeIpAddressArray,
      char separator);
    uint8_t* convertBinToHex(
      uint8_t* arrayBin, int sizeArrayBin);
    uint8_t* convertIntToBin(uint16_t value);
    
    /* = {0x45, // Version and Header Length
          0x00, // Differentiated Services Field
          0x00, 0x28, // Total length : 40
          0x02, 0x12, // Identification 
          0x00, // Flags
          0x00, // Fragment offset
          0xff, // Time to live 
          0x11, // UDP Protocol
          0xed, 0x7e, // Header Checksum
          0x0a, 0x2a, 0x00, 0x4e, // Source adress
          0xc0, 0xa8, 0x01, 0x14}; // Destination address*/

  public:
    // Constructor
    Ipv4Protocol();

    // Others
    void resetChecksum();
    void resetTotalLength();

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
    uint8_t getVersionProto();
    uint8_t getServiceField();
    uint8_t* getTotalLength();
    // ...
    uint8_t* getChecksum();
    

    // Setter
    void setVersionProto(uint8_t versionProto,
                         bool resetChecksum);
    void setServiceField(uint8_t serviceField,
                         bool resetChecksum);
    void setTotalLength(uint8_t *totalLength,
                        bool resetChecksum);
    void setIdentification(uint8_t *identification,
                           bool resetChecksum);
    void setFlags(uint8_t flags,
                  bool resetChecksum);
    void setFragmentOffset(uint8_t fragmentOffset,
                           bool resetChecksum);
    void setTtl(uint8_t ttl,
                bool resetChecksum);
    void setProtocol(uint8_t protocol,
                     bool resetChecksum);
    void setChecksum(uint8_t *checksum,
                     bool resetChecksum);
    void setSrcAddress(uint8_t *srcAddress,
                       bool resetChecksum);
    void setSrcAddress(String srcAddress, // Overload
                       int sizeAddress,
                       bool resetChecksum);
    void setDstAddress(uint8_t *dstAddress,
                       bool resetChecksum);
    void setDstAddress(String dstAddress,
                       int sizeAddress,
                       bool resetChecksum);
    void setHeader(uint8_t *header, 
                   bool resetChecksum);
    void setPayload(uint8_t *payload,
                    int sizePayload,
                    bool resetTotalLength);
    
};

#endif
