#include "udp_protocol.h"

/*
 * CONSTRUCTOR
 */

/*  0xc7, 0x1b, // Source port
    0x09, 0x56, // Destination port
    0x00, 0x14, // Length
    0xf0, 0xa5, // Checksum */
                           
UdpProtocol::UdpProtocol():
         _srcPort{0xc7, 0x1b},
         _dstPort{0x09, 0x56},
         //_length{0x00, 0x14},
         _length{0x00, 0x08},
         _checksum{0xf0, 0xa5}
{

  // Src port
  _header[UDP_SRC_PORT_P] = _srcPort[0];
  _header[UDP_SRC_PORT_P+1] = _srcPort[1];

  // Dst port
  _header[UDP_DST_PORT_P] = _dstPort[0];
  _header[UDP_DST_PORT_P+1] = _dstPort[1];

  // Length
  _header[UDP_LENGTH_P] = _length[0];
  _header[UDP_LENGTH_P+1] = _length[1];

  // Checksum
  _header[UDP_CHECKSUM_P] = _checksum[0];
  _header[UDP_CHECKSUM_P+1] = _checksum[1];

  // Define data size
  _sizePayload = 0;

};


/*
 * OTHERS
 */


// See : <bitset> for memory improvement
uint8_t* UdpProtocol::convertBinToHex(
                    uint8_t* arrayBin,
                    int sizeArrayBin)
{
  int sizeArrayHex = (int)sizeArrayBin / 8;
  uint8_t* arrayHex = new uint8_t[sizeArrayHex];
  
  int value = 0;
  int index = 0;

  for (int i=0; i<sizeArrayBin; i=i+8)
  {
    for (int j=0; j<8; j++)
    {
      value = value + pow(2, 7-j)*arrayBin[i+j];
    }
    arrayHex[index] = value;
    index++;
    value = 0;
  }
  return arrayHex;
}


uint8_t* UdpProtocol::convertIntToBin(uint16_t value)
{
  int sizeValue = sizeof(value)*8;
  uint8_t* arrayBin = new uint8_t[sizeValue];

  // Assume it's a 32 bit int
  for (int i=0; i<sizeValue; i++)
  {
    arrayBin[sizeValue-1-i] = (value >> i) & 1;
  }
  return arrayBin;
}


// FROM : https://github.com/renatoaloi/EtherEncLib/blob/master/checksum.c
uint16_t UdpProtocol::computeChecksum(
  uint8_t *buf, uint16_t len, uint8_t type)
{
        // type 0=ip 
        //      1=udp
        //      2=tcp
        uint32_t sum = 0;

        //if(type==0){
        //        // do not add anything
        //}
        if(type==1){ // compute with _protocol !!!
                sum+=UDP_PROTO_UDP_V; // protocol udp
                // the length here is the length of udp (data+header len)
                // =length given to this function - (IPv4.scr+IPv4.dst length)
                sum+=len;//-8; // = real tcp len
        }
        if(type==2){
                sum+=UDP_PROTO_TCP_V; 
                // the length here is the length of tcp (data+header len)
                // =length given to this function - (IPv4.scr+IPv4.dst length)
                sum+=len-8; // = real tcp len
        }
        // build the sum of 16bit words
        while(len >1){
                sum += 0xFFFF & (*buf<<8|*(buf+1));
                buf+=2;
                len-=2;
        }
        // if there is a byte left then add it (padded with zero)
        if (len){
        //--- made by SKA ---                sum += (0xFF & *buf)<<8;
                sum += 0xFFFF & (*buf<<8|0x00);
        }
        // now calculate the sum over the bytes in the sum
        // until the result is only 16bit long
        while (sum>>16){
                sum = (sum & 0xFFFF)+(sum >> 16);
        }
        // build 1's complement:
        return( (uint16_t) sum ^ 0xFFFF);
}


void UdpProtocol::resetChecksum()
{
  _header[UDP_CHECKSUM_P] = 0x00;
  _header[UDP_CHECKSUM_P+1] = 0x00;
  uint16_t ck = this->computeChecksum(
    &_header[0], UDP_HEADER_LEN_V, 0);
  _checksum[0] = ck >> 8;
  _checksum[1] = ck & 0xff;
  _header[UDP_CHECKSUM_P] = _checksum[0];
  _header[UDP_CHECKSUM_P+1] = _checksum[1];
}


void UdpProtocol::resetLength()
{
  uint16_t value = this->getSizeData();
  int sizeValueBin = sizeof(value)*8;
  uint8_t* valueBin = this->convertIntToBin(value);
  uint8_t* valueHex = this->convertBinToHex(
    valueBin, sizeValueBin);

  _length[0] = valueHex[0];
  _length[1] = valueHex[1];
  _header[UDP_LENGTH_P] = _length[0];
  _header[UDP_LENGTH_P+1] = _length[1];

  delete [] valueBin;
  delete [] valueHex;
}


/*
 * PRINTER
 */


void UdpProtocol::printHeader()
{
  Serial.print("[ "); 
  for(int i = 0; i < UDP_HEADER_LEN_V-1; i++) { 
    Serial.print(_header[i]);
    Serial.print(", ");
  }

  // Print last element
  Serial.print(_header[UDP_HEADER_LEN_V-1]);
  Serial.println(" ]");
}

void UdpProtocol::printPayload()
{
  Serial.print("[ "); 
  for(int i = 0; i < _sizePayload-1; i++) { 
    Serial.print(_payload[i]);
    Serial.print(", ");
  }

  // Print last element
  Serial.print(_payload[_sizePayload-1]);
  Serial.println(" ]");
}

void UdpProtocol::printData()
{
  uint8_t* data = this->getData();
  int sizeData = this->getSizeData();
  Serial.print("[ "); 
  for(int i = 0; i < sizeData-1; i++) { 
    Serial.print(data[i]);
    Serial.print(", ");
  }

  // Print last element
  Serial.print(data[sizeData-1]);
  Serial.println(" ]");
}


/*
 * GETTER
 */



uint8_t* UdpProtocol::getData()
{
  int sizeData = this->getSizeData();
  uint8_t* data = new uint8_t[sizeData];
  
  // Set header
  for(int i=0; i<UDP_HEADER_LEN_V; i++)
  {
    data[i] = _header[i];
  }

  // Set payload
  for(int i=UDP_HEADER_LEN_V; i<sizeData; i++)
  {
    data[i] = _payload[i-UDP_HEADER_LEN_V];
  }
  
  return data;
}

int UdpProtocol::getSizeData()
{
  int sizeData = UDP_HEADER_LEN_V+_sizePayload;
  return sizeData;
}

uint8_t* UdpProtocol::getHeader()
{
  return _header; 
}

uint8_t* UdpProtocol::getPayload()
{
  // return &_payload[0];
  return _payload;  
}

int UdpProtocol::getSizePayload()
{
  return _sizePayload; 
}

uint8_t* UdpProtocol::getSrcPort()
{
  return _srcPort; 
}

uint8_t* UdpProtocol::getDstPort()
{
  return _dstPort; 
}

uint8_t* UdpProtocol::getLength()
{
  return _length; // renvoie valeurs (que a la declaration que c'est un pointeurs) 
}

uint8_t* UdpProtocol::getChecksum()
{
  return _checksum; // renvoie valeurs (que a la declaration que c'est un pointeurs) 
}


/*
 * SETTER
 */


void UdpProtocol::setHeader(uint8_t *header,
                            bool resetChecksum=true)
{
  memcpy(_header, header, UDP_HEADER_LEN_V);
  if (resetChecksum) 
  {
    this->resetChecksum();
  }
}

void UdpProtocol::setPayload(uint8_t *payload,
                             int sizePayload,
                             bool resetChecksum=true,
                             bool resetLength=true)
{
  memcpy(_payload, payload, sizePayload);
  _sizePayload = sizePayload;
  if (resetLength)
  {
    this->resetLength();
  }
  if (resetChecksum) 
  {
    this->resetChecksum();  
  }
}

void UdpProtocol::setSrcPort(uint8_t* srcPort,
                             bool resetChecksum=true)
{
  memcpy(_srcPort, srcPort, 2);
  _header[UDP_SRC_PORT_P] = _srcPort[0];
  _header[UDP_SRC_PORT_P+1] = _srcPort[1];
  if (resetChecksum) 
  {
    this->resetChecksum();  
  } 
}

void UdpProtocol::setDstPort(uint8_t* dstPort,
                             bool resetChecksum=true)
{
  memcpy(_dstPort, dstPort, 2);
  _header[UDP_DST_PORT_P] = _dstPort[0];
  _header[UDP_DST_PORT_P+1] = _dstPort[1];
  if (resetChecksum) 
  {
    this->resetChecksum();  
  }
}

void UdpProtocol::setLength(uint8_t* dstPort,
                             bool resetChecksum=true)
{
  memcpy(_dstPort, dstPort, 2);
  _header[UDP_LENGTH_P] = _dstPort[0];
  _header[UDP_LENGTH_P+1] = _dstPort[1];
  if (resetChecksum) 
  {
    this->resetChecksum();  
  }
}

void UdpProtocol::setChecksum(uint8_t* checksum,
                              bool resetChecksum=true)
{
  if (resetChecksum) {
    this->resetChecksum();  
  } else {
    memcpy(_checksum, checksum, 2);
    _header[UDP_CHECKSUM_P] = _checksum[0];
    _header[UDP_CHECKSUM_P+1] = _checksum[1];
  }
}
