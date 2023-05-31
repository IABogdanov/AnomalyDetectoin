EMAG2_V3 Format File

EMAG2_V3_[yyyymmdd] is a Comma Delimited file that has the publication date appended to the file name to indicate update version.

Column 1: i ; grid column/longitude index
Column 2: j ; grid row/latitude index
Column 3: LON ; Geographic Longitude WGS84 (decimal degrees)
Column 4: LAT ; Geographic Latitude WGS84 (decimal degrees)
Column 5: SeaLevel ; Magnetic Anomaly Value at Sea Level(nT)
Column 6: UpCont ; Magnetic Anomaly Value at continuous 4km altitude (nT)
Column 7: Code ; Data Source Code (see table below)
Column 8: Error ; Error estimate (nT)

Source Code Table (Column 7 values):
001 West Asia
002 Antarctica
003 Afghanistan
004 Alaska
005 Algeria
006 Arctic
007 Argentina onshore
008 Argentina offshore
009 Australia
010 Austria
011 Bolivia
013 East Asia
014 Europe
015 Eurasia
016 Fennoscandia
017 France
018 Western Europe
019 Getech
020 India
021 East India
022 Iraq
023 Italy
024 Ivory Coast
025 Japan
026 Hawaii
027 Middle East
029 New Zealand
030 Pakistan
031 Papua
032 Russia
033 Saudi Arabia
034 South Africa
035 Spain
036 Tanzania
037 NAMAG
038 Former Soviet Union
101 Arabian
102 Arabian aeromag
103 Baja
104 Central Atlantic
105 Caribbean
106 Fuego
107 Hope
108 Indian
109 Mediterranean
110 North Atlantic 1
111 North Atlantic 2
112 North Atlantic 3
113 North Atlantic 4
114 North Atlantic 5
115 North Pacific
116 Northwest Pacific
117 South Ocean
118 South Pacific
119 Southeast Pacific
120 Southwest Atlantic
121 Tasman
888 Ambiguous
999 No data

Code 888 is assigned in certain cells on grid edges where the data source is ambiguous and assigned an error of -888 nT.
Code 999 is assigned in cells where no data is reported with the anomaly value assigned 99999 nT and an error of -999 nT.
