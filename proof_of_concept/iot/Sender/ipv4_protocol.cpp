#include "ipv4_protocol.h"


/*
 * CONSTRUCTOR
 */

                           
Ipv4Protocol::Ipv4Protocol():
         _versionProto(0x45),
         _serviceField(0x00),
         _totalLength{0x00, 0x28}, // Header + payload to compute ?
         _identification{0x02, 0x12},
         _flags(0x00),
         _fragmentOffset(0x00),
         _ttl(0xff),
         _protocol(0x11),
         _checksum{0xed, 0x7e},
         _srcAddress{0x00, 0x00, 0x00, 0x00},
         _dstAddress{0xff, 0xff, 0xff, 0xff}
{

  // Version Protocol
  _header[IPv4_VERSION_P] = _versionProto;

  // Service Field
  _header[IPv4_SERVICE_P] = _serviceField;

  // Total Length
  _header[IPv4_LENGTH_P] = _totalLength[0];
  _header[IPv4_LENGTH_P+1] = _totalLength[1];

  // Identification
  _header[IPv4_ID_P] = _identification[0];
  _header[IPv4_ID_P+1] = _identification[1];

  // Flags
  _header[IPv4_FLAGS_P] = _flags;

  // Fragment Offset
  _header[IPv4_FRAGMENT_P] = _fragmentOffset;

  // TTL
  _header[IPv4_TTL_P] = _ttl;

  // Protocol
  _header[IPv4_PROTOCOL_P] = _protocol;

  // Src address
  for(int i=IPv4_SRC_P; i<IPv4_SRC_P+4; i++)
  {
    _header[i] = _srcAddress[i-IPv4_SRC_P];
  }

  // Dst address
  for(int i=IPv4_DST_P; i<IPv4_DST_P+4; i++)
  {
    _header[i] = _dstAddress[i-IPv4_DST_P];
  }

  // Compute checksum
  this->resetChecksum();

  // Define header size
  _sizePayload = 0;

};



/*
 * OTHERS
 */

// See : <bitset> for memory improvement
uint8_t* Ipv4Protocol::convertBinToHex(
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


uint8_t* Ipv4Protocol::convertIntToBin(uint16_t value)
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
uint16_t Ipv4Protocol::computeChecksum(uint8_t *buf, uint16_t len, uint8_t type)
{
    // type 0=ip 
    //      1=udp
    //      2=tcp
    uint32_t sum = 0;

    //if(type==0){
    //        // do not add anything
    //}
    if(type==1){ // compute with _protocol !!!
            sum+=IPv4_PROTO_UDP_V; // protocol udp
            // the length here is the length of udp (data+header len)
            // =length given to this function - (IPv4.scr+IPv4.dst length)
            sum+=len-8; // = real tcp len
    }
    if(type==2){
            sum+=IPv4_PROTO_TCP_V; 
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


void Ipv4Protocol::resetChecksum()
{
  _header[IPv4_CHECKSUM_P] = 0x00;
  _header[IPv4_CHECKSUM_P+1] = 0x00;
  uint16_t ck = this->computeChecksum(&_header[0], IPv4_HEADER_LEN_V, 0);
  _checksum[0] = ck >> 8;
  _checksum[1] = ck & 0xff;
  _header[IPv4_CHECKSUM_P] = _checksum[0];
  _header[IPv4_CHECKSUM_P+1] = _checksum[1];
}


void Ipv4Protocol::resetTotalLength()
{
  uint16_t value = this->getSizeData();
  int sizeValueBin = sizeof(value)*8;
  uint8_t* valueBin = this->convertIntToBin(value);
  uint8_t* valueHex = this->convertBinToHex(
    valueBin, sizeValueBin);

  _totalLength[0] = valueHex[0];
  _totalLength[1] = valueHex[1];
  _header[IPv4_LENGTH_P] = _totalLength[0];
  _header[IPv4_LENGTH_P+1] = _totalLength[1];

  delete [] valueBin;
  delete [] valueHex;
}


uint8_t* Ipv4Protocol::ipAddressStringToHex(
  String ipAddressString, 
  int sizeIpAddressArray,
  char separator='.')
{
  char ipAddressArray[16]; // Ipv6 length by default
  uint8_t* ipAddressHexArray = new uint8_t[4];

  // Convert string to array char
  strcpy(ipAddressArray, ipAddressString.c_str());
  //sizeIpAddressArray = sizeof(ipAddressArray)/sizeof(ipAddressArray[0]);

  int counter = 0;
  int idxArray = 0;
  for (int i=0; i<=sizeIpAddressArray; i++) 
  {
    char valAddr = ipAddressArray[i];
    if (!strncmp(&separator, &valAddr, 1) || (i == sizeIpAddressArray))
    {
      String s = "";
      for (int j=0; j<counter; j++)
      {
        // Create String
        s = s + ipAddressArray[(i-counter)+j];
      }
      
      // Convert to hexa
      ipAddressHexArray[idxArray] = atoi(s.c_str());
      
      counter = 0;
      idxArray++;
    } else {
      counter++;
    }
  }
  return ipAddressHexArray; 
}



/*
 * PRINTER
 */

void Ipv4Protocol::printHeader()
{
  Serial.print("[ "); 
  for(int i = 0; i < IPv4_HEADER_LEN_V-1; i++) { 
    Serial.print(_header[i]);
    Serial.print(", ");
  }

  // Print last element
  Serial.print(_header[IPv4_HEADER_LEN_V-1]);
  Serial.println(" ]");
}

void Ipv4Protocol::printPayload()
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

void Ipv4Protocol::printData()
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
 

uint8_t* Ipv4Protocol::getData()
{
  int sizeData = this->getSizeData();
  uint8_t* data = new uint8_t[sizeData];
  
  // Set header
  for(int i=0; i<IPv4_HEADER_LEN_V; i++)
  {
    data[i] = _header[i];
  }

  // Set payload
  for(int i=IPv4_HEADER_LEN_V; i<=sizeData; i++)
  {
    data[i] = _payload[i-IPv4_HEADER_LEN_V];
  }
  
  return &data[0];
}

int Ipv4Protocol::getSizeData()
{
  int sizeData = IPv4_HEADER_LEN_V+_sizePayload;
  return sizeData;
}

uint8_t* Ipv4Protocol::getHeader() 
{
  return _header; 
}

int Ipv4Protocol::getSizeHeader() 
{
  return IPv4_HEADER_LEN_V; 
}

uint8_t* Ipv4Protocol::getPayload() 
{
  return _payload; 
}

int Ipv4Protocol::getSizePayload() 
{
  return _sizePayload; 
}

uint8_t Ipv4Protocol::getVersionProto()
{
  return _versionProto; 
}

uint8_t Ipv4Protocol::getServiceField()
{
  return _serviceField; 
}

uint8_t* Ipv4Protocol::getTotalLength()
{
  return _totalLength;
}

// ...

uint8_t* Ipv4Protocol::getChecksum()
{
  return _checksum; // idem as : &_checksum[0];
}


/*
 * SETTER
 */


void Ipv4Protocol::setHeader(uint8_t *header,
                             bool resetChecksum=true)
{
  memcpy(_header, header, IPv4_HEADER_LEN_V);
  if (resetChecksum) 
  {
    this->resetChecksum();  
  }
}

void Ipv4Protocol::setPayload(uint8_t *payload,
                              int sizePayload,
                              bool resetTotalLength=true)
{
  _sizePayload = sizePayload;
  if (resetTotalLength)
  {
    this->resetTotalLength();
  }
  memcpy(_payload, payload, _sizePayload);
}

void Ipv4Protocol::setVersionProto(uint8_t versionProto,
                                   bool resetChecksum=true)
{
  _versionProto = versionProto;
  _header[IPv4_VERSION_P] = _versionProto;
  if (resetChecksum) 
  {
    this->resetChecksum();  
  }
}

void Ipv4Protocol::setServiceField(uint8_t serviceField,
                                   bool resetChecksum=true)
{
  _serviceField = serviceField;
  _header[IPv4_SERVICE_P] = _serviceField;
  if (resetChecksum) 
  {
    this->resetChecksum();  
  }
}

void Ipv4Protocol::setTotalLength(uint8_t* totalLength,
                                  bool resetChecksum=true)
                                  
{
  memcpy(_totalLength, totalLength, 2);
  _header[IPv4_LENGTH_P] = _totalLength[0];
  _header[IPv4_LENGTH_P+1] = _totalLength[1];
  if (resetChecksum) 
  {
    this->resetChecksum();  
  }
}

void Ipv4Protocol::setIdentification(uint8_t* identification,
                                     bool resetChecksum=true)
{
  memcpy(_identification, identification, 2);
  _header[IPv4_ID_P] = _identification[0];
  _header[IPv4_ID_P+1] = _identification[1];
  if (resetChecksum) 
  {
    this->resetChecksum();  
  }
}

void Ipv4Protocol::setFlags(uint8_t flags,
                            bool resetChecksum=true)
{
  _flags = flags;
  _header[IPv4_FLAGS_P] = _flags;
  if (resetChecksum) 
  {
    this->resetChecksum();  
  } 
}

void Ipv4Protocol::setFragmentOffset(uint8_t fragmentOffset,
                                     bool resetChecksum=true)
{
  _fragmentOffset = fragmentOffset;
  _header[IPv4_FRAGMENT_P] = _fragmentOffset;
  if (resetChecksum) 
  {
    this->resetChecksum();  
  } 
}

void Ipv4Protocol::setTtl(uint8_t ttl,
                          bool resetChecksum=true)
{
  _ttl = ttl;
  _header[IPv4_TTL_P] = _ttl;
  if (resetChecksum) 
  {
    this->resetChecksum();  
  }
}

void Ipv4Protocol::setProtocol(uint8_t protocol,
                               bool resetChecksum=true)
{
  _protocol = protocol;
  _header[IPv4_TTL_P] = _ttl;
  if (resetChecksum) 
  {
    this->resetChecksum();  
  }
}

void Ipv4Protocol::setChecksum(uint8_t* checksum,
                               bool resetChecksum=true)
{
  if (resetChecksum) {
    this->resetChecksum();  
  } else {
    memcpy(_checksum, checksum, 2);
    _header[IPv4_CHECKSUM_P] = _checksum[0];
    _header[IPv4_CHECKSUM_P+1] = _checksum[1];
  }
}

void Ipv4Protocol::setSrcAddress(uint8_t* srcAddress,
                                 bool resetChecksum=true)
{
  memcpy(_srcAddress, srcAddress, 4);
  for(int i=IPv4_SRC_P; i<IPv4_SRC_P+4; i++)
  {
    _header[i] = _srcAddress[i-IPv4_SRC_P];
  }
  
  if (resetChecksum) 
  {
    this->resetChecksum();  
  }
}

void Ipv4Protocol::setSrcAddress(String srcAddress,
                                 int sizeAddress,
                                 bool resetChecksum=true)
{
  printf("[DEBUG][Ipv4Protocol::setSrcAddress] In the function\n");
  uint8_t* srcAddressHex = this->ipAddressStringToHex(
    srcAddress, sizeAddress, '.');
  memcpy(_srcAddress, srcAddressHex, 4);
  for(int i=IPv4_SRC_P; i<IPv4_SRC_P+4; i++)
  {
    _header[i] = _srcAddress[i-IPv4_SRC_P];
  }
  
  if (resetChecksum) 
  {
    this->resetChecksum();  
  }
}

void Ipv4Protocol::setDstAddress(uint8_t* dstAddress,
                                 bool resetChecksum=true)
{
  memcpy(_dstAddress, dstAddress, 4);
  for(int i=IPv4_DST_P; i<IPv4_DST_P+4; i++)
  {
    _header[i] = _dstAddress[i-IPv4_DST_P];
  }
  
  if (resetChecksum) 
  {
    this->resetChecksum();  
  }
}

void Ipv4Protocol::setDstAddress(String dstAddress,
                                 int sizeAddress,
                                 bool resetChecksum=true)
{
  uint8_t* dstAddressHex = this->ipAddressStringToHex(
    dstAddress, sizeAddress, '.');
  memcpy(_dstAddress, dstAddressHex, 4);
  for(int i=IPv4_DST_P; i<IPv4_DST_P+4; i++)
  {
    _header[i] = _dstAddress[i-IPv4_DST_P];
  }
  
  if (resetChecksum) 
  {
    this->resetChecksum();  
  }
}
