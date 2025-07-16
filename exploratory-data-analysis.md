---
language: en
---

# Toronto KSI Collisions Dataset

## 1. Dataset Description

This dataset contains information on Killed or Seriously Injured (KSI) collisions that occurred in Toronto, sourced from the Toronto Police Service Public Safety Data Portal. It provides detailed records of traffic accidents, including location, time, contributing factors, and involved parties, with a focus on incidents resulting in fatalities or serious injuries.

## 2. Data Fields

This section describes the various fields present in the dataset used for this project, including information on data types and the count of missing values.

| Field | Field Name          | Description                                                                        | Categorical | Categories (if applicable)                                            | Missing/Null Count | Data Type |
| ----- | ------------------- | ---------------------------------------------------------------------------------- | ----------- | --------------------------------------------------------------------- | ------------------ | --------- |
| 0     | `OBJECTID`          | Unique Identifier (auto generated)                                                 | No          | N/A                                                                   | 0                  | `int64`   |
| 1     | `INDEX_`            | Unique Identifier                                                                  | No          | N/A                                                                   | 0                  | `int64`   |
| 2     | `ACCNUM`            | Accident Number                                                                    | No          | N/A                                                                   | 4930               | `float64` |
| 3     | `DATE`              | Date Collision Occurred (time is displayed in UTC format when downloaded as a CSV) | No          | N/A                                                                   | 0                  | `object`  |
| 4     | `TIME`              | Time Collision Occurred                                                            | No          | N/A                                                                   | 0                  | `int64`   |
| 5     | `STREET1`           | Street Collision Occurred                                                          | Yes         | Varies (Street Names)                                                 | 0                  | `object`  |
| 6     | `STREET2`           | Street Collision Occurred                                                          | Yes         | Varies (Street Names)                                                 | 1706               | `object`  |
| 7     | `OFFSET`            | Distance and direction of the Collision                                            | No          | N/A                                                                   | 15137              | `object`  |
| 8     | `ROAD_CLASS`        | Road Classification                                                                | Yes         | e.g., Major Arterial, Collector, Local, Expressway                    | 486                | `object`  |
| 9     | `DISTRICT`          | City District                                                                      | Yes         | e.g., Etobicoke, North York, Scarborough, Toronto, York               | 229                | `object`  |
| 10    | `LATITUDE`          | Latitude                                                                           | No          | N/A                                                                   | 0                  | `float64` |
| 11    | `LONGITUDE`         | Longitude                                                                          | No          | N/A                                                                   | 0                  | `float64` |
| 12    | `ACCLOC`            | Collision Location                                                                 | Yes         | e.g., Intersection, Mid-Block                                         | 5456               | `object`  |
| 13    | `TRAFFCTL`          | Traffic Control Type                                                               | Yes         | e.g., Traffic Signal, Stop Sign, No Control, Pedestrian Crossover     | 75                 | `object`  |
| 14    | `VISIBILITY`        | Environment Condition                                                              | Yes         | e.g., Clear, Rain, Snow, Fog                                          | 24                 | `object`  |
| 15    | `LIGHT`             | Light Condition                                                                    | Yes         | e.g., Day, Dusk, Dawn, Dark (no street lights), Dark (street lights)  | 4                  | `object`  |
| 16    | `RDSFCOND`          | Road Surface Condition                                                             | Yes         | e.g., Dry, Wet, Snow, Ice                                             | 29                 | `object`  |
| 17    | `ACCLASS`           | Classification of Accident                                                         | Yes         | e.g., Fatal, Non-Fatal Injury, Property Damage Only                   | 1                  | `object`  |
| 18    | `IMPACTYPE`         | Initial Impact Type                                                                | Yes         | e.g., Rear-End, Side-Impact, Head-On, Single Vehicle                  | 27                 | `object`  |
| 19    | `INVTYPE`           | Involvement Type                                                                   | Yes         | e.g., Driver, Pedestrian, Cyclist, Passenger                          | 16                 | `object`  |
| 20    | `INVAGE`            | Age of Involved Party                                                              | No          | N/A (though often grouped into categories for analysis)               | 0                  | `object`  |
| 21    | `INJURY`            | Severity of Injury                                                                 | Yes         | e.g., Fatal, Serious, Minor, No Injury                                | 8897               | `object`  |
| 22    | `FATAL_NO`          | Sequential Number                                                                  | No          | N/A                                                                   | 18087              | `float64` |
| 23    | `INITDIR`           | Initial Direction of Travel                                                        | Yes         | e.g., North, South, East, West                                        | 5277               | `object`  |
| 24    | `VEHTYPE`           | Type of Vehicle                                                                    | Yes         | e.g., Passenger Car, Truck, Motorcycle, Bus, Bicycle                  | 3487               | `object`  |
| 25    | `MANOEUVER`         | Vehicle Manoeuver                                                                  | Yes         | e.g., Going Ahead, Turning Left, Turning Right, Changing Lanes        | 7953               | `object`  |
| 26    | `DRIVACT`           | Apparent Driver Action                                                             | Yes         | e.g., Impaired, Disobeyed Traffic Sign, Speeding, Distracted          | 9289               | `object`  |
| 27    | `DRIVCOND`          | Driver Condition                                                                   | Yes         | e.g., Normal, Fatigued, Impaired, Medical Condition                   | 9291               | `object`  |
| 28    | `PEDTYPE`           | Pedestrian Crash Type - detail                                                     | Yes         | Varies (Specific pedestrian involvement scenarios)                    | 15728              | `object`  |
| 29    | `PEDACT`            | Pedestrian Action                                                                  | Yes         | e.g., Crossing with right-of-way, Crossing against signal, Jaywalking | 15730              | `object`  |
| 30    | `PEDCOND`           | Condition of Pedestrian                                                            | Yes         | e.g., Normal, Impaired, Medical Condition                             | 15711              | `object`  |
| 31    | `CYCLISTYPE`        | Cyclist Crash Type - detail                                                        | Yes         | Varies (Specific cyclist involvement scenarios)                       | 18152              | `object`  |
| 32    | `CYCACT`            | Cyclist Action                                                                     | Yes         | e.g., Going Ahead, Turning, Changing Lanes, Unknown                   | 18155              | `object`  |
| 33    | `CYCCOND`           | Cyclist Condition                                                                  | Yes         | e.g., Normal, Impaired, Medical Condition                             | 18157              | `object`  |
| 34    | `PEDESTRIAN`        | Pedestrian Involved In Collision                                                   | Yes         | Boolean (Yes/No, 1/0)                                                 | 11269              | `object`  |
| 35    | `CYCLIST`           | Cyclists Involved in Collision                                                     | Yes         | Boolean (Yes/No, 1/0)                                                 | 16971              | `object`  |
| 36    | `AUTOMOBILE`        | Driver Involved in Collision                                                       | Yes         | Boolean (Yes/No, 1/0)                                                 | 1727               | `object`  |
| 37    | `MOTORCYCLE`        | Motorcyclist Involved in Collision                                                 | Yes         | Boolean (Yes/No, 1/0)                                                 | 17273              | `object`  |
| 38    | `TRUCK`             | Truck Driver Involved in Collision                                                 | Yes         | Boolean (Yes/No, 1/0)                                                 | 17788              | `object`  |
| 39    | `TRSN_CITY_VEH`     | Transit or City Vehicle Involved in Collision                                      | Yes         | Boolean (Yes/No, 1/0)                                                 | 17809              | `object`  |
| 40    | `EMERG_VEH`         | Emergency Vehicle Involved in Collision                                            | Yes         | Boolean (Yes/No, 1/0)                                                 | 18908              | `object`  |
| 41    | `PASSENGER`         | Passenger Involved in Collision                                                    | Yes         | Boolean (Yes/No, 1/0)                                                 | 11774              | `object`  |
| 42    | `SPEEDING`          | Speeding Related Collision                                                         | Yes         | Boolean (Yes/No, 1/0)                                                 | 16263              | `object`  |
| 43    | `AG_DRIV`           | Aggressive and Distracted Driving Collision                                        | Yes         | Boolean (Yes/No, 1/0)                                                 | 9121               | `object`  |
| 44    | `REDLIGHT`          | Red Light Related Collision                                                        | Yes         | Boolean (Yes/No, 1/0)                                                 | 17380              | `object`  |
| 45    | `ALCOHOL`           | Alcohol Related Collision                                                          | Yes         | Boolean (Yes/No, 1/0)                                                 | 18149              | `object`  |
| 46    | `DISABILITY`        | Medical or Physical Disability Related Collision                                   | Yes         | Boolean (Yes/No, 1/0)                                                 | 18464              | `object`  |
| 47    | `HOOD_158`          | Unique ID for City of Toronto Neighbourhood (new)                                  | Yes         | Numeric IDs, representing specific neighborhoods                      | 0                  | `object`  |
| 48    | `NEIGHBOURHOOD_158` | City of Toronto Neighbourhood name (new)                                           | Yes         | Varies (Specific neighborhood names)                                  | 0                  | `object`  |
| 49    | `HOOD_140`          | Unique ID for City of Toronto Neighbourhood (old)                                  | Yes         | Numeric IDs, representing specific neighborhoods                      | 0                  | `object`  |
| 50    | `NEIGHBOURHOOD_140` | City of Toronto Neighbourhood name (old)                                           | Yes         | Varies (Specific neighborhood names)                                  | 0                  | `object`  |
| 51    | `DIVISION`          | Toronto Police Service Division                                                    | Yes         | e.g., 51 Division, 52 Division, 53 Division, etc.                     | 0                  | `object`  |
| 52    | `x`                 | X Coordinate (likely related to Latitude/Longitude)                                | No          | N/A                                                                   | 0                  | `float64` |
| 53    | `y`                 | Y Coordinate (likely related to Latitude/Longitude)                                | No          | N/A                                                                   | 0                  | `float64` |

This section describes the various fields present in the dataset used for this project.
_(Add any other columns you have, especially if you created new features during preprocessing, e.g., `DAY_OF_WEEK`, `HOUR_OF_DAY`, `IS_WEEKEND`)_

## 3. Data Stats

### Numerical Features

| Feature Name | Count   | Mean      | Std Dev  | Min Value | 25th Percentile | 50th Percentile (Median) | 75th Percentile | Max Value | Notes                                                                                   |
| ------------ | ------- | --------- | -------- | --------- | --------------- | ------------------------ | --------------- | --------- | --------------------------------------------------------------------------------------- |
| `OBJECTID`   | 18957.0 | 9479.0    | 5472.56  | 1.0       | 4740.0          | 9479.0                   | 14218.0         | 18957.0   | Unique identifier; evenly distributed.                                                  |
| `INDEX_`     | 18957.0 | 3.99e+07  | 3.75e+07 | 3.36e+06  | 5.41e+06        | 7.82e+06                 | 8.09e+07        | 8.18e+07  | Another unique identifier; wide range.                                                  |
| `ACCNUM`     | 14027.0 | 5.58e+08  | 1.18e+09 | 2.53e+04  | 1.03e+06        | 1.22e+06                 | 1.39e+06        | 4.01e+09  | Accident number; significant missing values (4930).                                     |
| `TIME`       | 18957.0 | 1364.96   | 631.31   | 0.0       | 924.0           | 1450.0                   | 1852.0          | 2359.0    | Represents time in HHMM format. Distribution suggests peaks around midday/afternoon.    |
| `LATITUDE`   | 18957.0 | 43.71     | 0.056    | 43.59     | 43.66           | 43.70                    | 43.76           | 43.86     | Geographic coordinate within Toronto.                                                   |
| `LONGITUDE`  | 18957.0 | -79.396   | 0.104    | -79.64    | -79.47          | -79.397                  | -79.318         | -79.123   | Geographic coordinate within Toronto.                                                   |
| `FATAL_NO`   | 870.0   | 28.75     | 17.66    | 1.0       | 14.0            | 27.5                     | 42.0            | 78.0      | Sequential number for fatalities; highly sparse (18087 missing). Can be used as a flag. |
| `x`          | 18957.0 | 629181.57 | 8364.34  | 609625.70 | 623177.00       | 629199.08                | 635424.04       | 651024.09 | Cartesian coordinate; likely derived from Lat/Long.                                     |
| `y`          | 18957.0 | 4.84e+06  | 6.32e+03 | 4.83e+06  | 4.84e+06        | 4.84e+06                 | 4.85e+06        | 4.86e+06  | Cartesian coordinate; likely derived from Lat/Long.                                     |

### Categorical Features

#### **Time and Location Descriptors**

- **`DATE`** (Top 5 Dates with most collisions):

  - 8/17/2014 8:00:00 AM: 35
  - 9/1/2007 8:00:00 AM: 24
  - 7/20/2012 8:00:00 AM: 23
  - 3/20/2016 8:00:00 AM: 22
  - 4/17/2007 8:00:00 AM: 22
  - _Note: Specific dates with high counts might indicate local events or data anomalies._

- **`STREET1`** (Top 5 primary streets):

  - YONGE ST: 403
  - BATHURST ST: 338
  - DUNDAS ST W: 304
  - DUFFERIN ST: 294
  - EGLINTON AVE E: 288
  - _Note: Identifies frequently occurring collision locations._

- **`STREET2`** (Top 5 secondary streets for collisions):

  - BATHURST ST: 156
  - LAWRENCE AVE E: 153
  - YONGE ST: 133
  - FINCH AVE E: 126
  - EGLINTON AVE E: 123
  - _Note: Complements STREET1 for intersection analysis._

- **`OFFSET`** (Top 5 offsets from a reference point):

  - 10 m West of: 61
  - 10 m North o: 60
  - 5 m South of: 60
  - 10 m South o: 59
  - 5 m East of: 55
  - _Note: Highly granular, may be challenging to use directly without aggregation or conversion to numerical distance._

- **`ROAD_CLASS`** (Road classification where collision occurred):

  - Major Arterial: 13376
  - Minor Arterial: 2958
  - Collector: 1032
  - Local: 865
  - Expressway: 164
  - _Note: Major arterials account for the vast majority of collisions._

- **`DISTRICT`** (City district where collision occurred):

  - Toronto and East York: 6328
  - Etobicoke York: 4342
  - Scarborough: 4270
  - North York: 3788
  - _Note: Provides geographical segmentation within Toronto._

- **`ACCLOC`** (Collision location type):

  - At Intersection: 8774
  - Non Intersection: 2660
  - Intersection Related: 1604
  - At/Near Private Drive: 407
  - Overpass or Bridge: 14
  - _Note: Intersections are the most common collision location. Significant missing values present._

- **`HOOD_158`** (Unique ID for new neighbourhood definition - Top 5 IDs):

  - 1: 597
  - 170: 376
  - 119: 361
  - 70: 353
  - 85: 304

- **`NEIGHBOURHOOD_158`** (City of Toronto Neighbourhood name (new) - Top 5 names):

  - West Humber-Clairville: 597
  - Yonge-Bay Corridor: 376
  - Wexford/Maryvale: 361
  - South Riverdale: 353
  - South Parkdale: 304
  - _Note: Provides finer-grained location information than District._

- **`HOOD_140`** (Unique ID for old neighbourhood definition - Top 5 IDs):

  - 77: 740
  - 1: 592
  - 76: 445
  - 137: 409
  - 119: 361

- **`NEIGHBOURHOOD_140`** (City of Toronto Neighbourhood name (old) - Top 5 names):

  - Waterfront Communities-The Island (77): 740
  - West Humber-Clairville (1): 592
  - Bay Street Corridor (76): 445
  - Woburn (137): 409
  - Wexford/Maryvale (119): 361
  - _Note: Provides alternative neighbourhood definitions; check for overlap/redundancy with \_158 versions._

- **`DIVISION`** (Toronto Police Service Division - Top 5 divisions):
  - D42: 1813
  - D55: 1530
  - D41: 1435
  - D22: 1413
  - D32: 1328
  - _Note: Another geographical indicator, potentially useful for resource allocation or local policy analysis._

### **Environmental Conditions**

- **`TRAFFCTL`** (Traffic Control Type):

  - No Control: 9021
  - Traffic Signal: 8035
  - Stop Sign: 1464
  - Pedestrian Crossover: 208
  - Traffic Controller: 108
  - _Note: 'No Control' and 'Traffic Signal' are the dominant categories._

- **`VISIBILITY`** (Environment Condition):

  - Clear: 16373
  - Rain: 1976
  - Snow: 356
  - Other: 98
  - Fog, Mist, Smoke, Dust: 52
  - _Note: Most collisions occur under clear visibility._

- **`LIGHT`** (Light Condition):

  - Daylight: 10779
  - Dark: 3746
  - Dark, artificial: 3552
  - Dusk: 253
  - Dusk, artificial: 253
  - _Note: Majority occur in daylight, but a significant portion in dark conditions, with or without artificial light._

- **`RDSFCOND`** (Road Surface Condition):
  - Dry: 15231
  - Wet: 3140
  - Loose Snow: 174
  - Other: 147
  - Slush: 102
  - _Note: Most collisions on dry surfaces, followed by wet surfaces._

### **Collision Characteristics**

- **`ACCLASS`** (Classification of Accident - **Your Target Variable**):

  - Non-Fatal Injury: 16268
  - Fatal: 2670
  - Property Damage O: 18
  - _Note: This is the primary target for your prediction. Observe the significant class imbalance between Non-Fatal Injury and Fatal collisions._

- **`IMPACTYPE`** (Initial Impact Type):
  - Pedestrian Collisions: 7684
  - Turning Movement: 2934
  - Cyclist Collisions: 1861
  - Rear End: 1804
  - SMV Other: 1465
  - _Note: Pedestrian collisions are the most frequent impact type._

### **Involved Party Details**

- **`INVTYPE`** (Involvement Type):

  - Driver: 8651
  - Pedestrian: 3275
  - Passenger: 2889
  - Vehicle Owner: 1638
  - Cyclist: 822
  - _Note: Drivers are the most frequently involved party._

- **`INVAGE`** (Age of Involved Party - Top 5 age ranges):

  - unknown: 2625
  - 20 to 24: 1800
  - 25 to 29: 1723
  - 30 to 34: 1450
  - 35 to 39: 1382
  - _Note: Contains 'unknown' values which need handling. This is a categorical representation of age._

- **`INJURY`** (Severity of Injury):

  - Major: 6445
  - Minor: 1479
  - Minimal: 1160
  - Fatal: 976
  - _Note: This is closely related to your target variable `ACCLASS`. The count for 'Fatal' here (976) differs from `ACCLASS` 'Fatal' (2670), which suggests careful reconciliation is needed for your target definition._

- **`INITDIR`** (Initial Direction of Travel):

  - East: 3388
  - West: 3339
  - South: 3226
  - North: 3201
  - Unknown: 526
  - _Note: Relatively even distribution across cardinal directions, with some 'Unknown' values._

- **`VEHTYPE`** (Type of Vehicle):

  - Automobile, Station Wagon: 7805
  - Other: 4753
  - Bicycle: 819
  - Motorcycle: 747
  - Municipal Transit Bus (TTC): 284
  - _Note: Automobiles are the most common vehicle type involved._

- **`MANOEUVER`** (Vehicle Manoeuver):

  - Going Ahead: 6542
  - Turning Left: 1877
  - Stopped: 634
  - Turning Right: 505
  - Slowing or Stopping: 294
  - _Note: 'Going Ahead' is by far the most frequent maneuver. Significant missing values present._

- **`DRIVACT`** (Apparent Driver Action):

  - Driving Properly: 4425
  - Failed to Yield Right of Way: 1603
  - Lost control: 1007
  - Improper Turn: 614
  - Other: 529
  - _Note: 'Driving Properly' is the top action, which might seem counter-intuitive for an accident; this category could include situations where the other party was at fault, or where the 'action' leading to the collision isn't well captured._

- **`DRIVCOND`** (Driver Condition):

  - Normal: 6158
  - Inattentive: 1603
  - Unknown: 1129
  - Medical or Physical Disability: 181
  - Had Been Drinking: 166
  - _Note: Majority are 'Normal', but 'Inattentive' is a significant factor. Also contains 'Unknown'._

- **`PEDTYPE`** (Pedestrian Crash Type - detail - Top 5):

  - Pedestrian hit at mid-block: 810
  - Vehicle turns left while ped crosses with ROW at inter.: 646
  - Vehicle is going straight thru inter.while ped cross without ROW: 543
  - Pedestrian hit on sidewalk or shoulder: 257
  - Vehicle is going straight thru inter.while ped cross with ROW: 189
  - _Note: Highly detailed, applies only when a pedestrian is involved. Very high missing value count for records without pedestrians._

- **`PEDACT`** (Pedestrian Action - Top 5):

  - Crossing with right of way: 1019
  - Crossing, no Traffic Control: 730
  - Crossing without right of way: 453
  - On Sidewalk or Shoulder: 284
  - Other: 240
  - _Note: Similar to PEDTYPE, applicable only to pedestrian-involved collisions._

- **`PEDCOND`** (Condition of Pedestrian - Top 5):

  - Normal: 1842
  - Inattentive: 560
  - Unknown: 399
  - Had Been Drinking: 222
  - Other: 84
  - _Note: Condition of pedestrian at the time of collision._

- **`CYCLISTYPE`** (Cyclist Crash Type - detail - Top 5):

  - Motorist turned left across cyclists path.: 140
  - Cyclist without ROW rides into path of motorist at inter, lnwy, dwy-Cyclist not turn.: 117
  - Cyclist and Driver travelling in same direction. One vehicle sideswipes the other.: 114
  - Cyclist and Driver travelling in same direction. One vehicle rear-ended the other.: 59
  - Motorist without ROW drives into path of cyclist at inter, lnwy, dwy-Driver not turn.: 58
  - _Note: Highly detailed, applies only when a cyclist is involved. Very high missing value count._

- **`CYCACT`** (Cyclist Action - Top 5):

  - Driving Properly: 444
  - Disobeyed Traffic Control: 85
  - Other: 79
  - Failed to Yield Right of Way: 65
  - Lost control: 42
  - _Note: Actions of the cyclist at the time of collision._

- **`CYCCOND`** (Condition of Cyclist - Top 5):
  - Normal: 557
  - Inattentive: 110
  - Unknown: 79
  - Had Been Drinking: 30
  - Other: 12
  - _Note: Condition of the cyclist at the time of collision._

### **Involvement Flags (Binary 'Yes' Features)**

- **`PEDESTRIAN`**: Yes: 7688
- **`CYCLIST`**: Yes: 1986
- **`AUTOMOBILE`**: Yes: 17230
- **`MOTORCYCLE`**: Yes: 1684
- **`TRUCK`**: Yes: 1169
- **`TRSN_CITY_VEH`**: Yes: 1148
- **`EMERG_VEH`**: Yes: 49
  - _Note: Very low count for emergency vehicles, indicating rarity. Consider if this feature is useful or too sparse._
- **`PASSENGER`**: Yes: 7183

### **Contributing Factors (Binary 'Yes' Features)**

- **`SPEEDING`**: Yes: 2694
- **`AG_DRIV`** (Aggressive and Distracted Driving Collision): Yes: 9836
- **`REDLIGHT`**: Yes: 1577
- **`ALCOHOL`**: Yes: 808
- **`DISABILITY`**: Yes: 493

## 4. Data Preprocessing and Feature Engineering

### Handling Missing Values

### Data Type Conversions

### Feature Creation (e.g., `DAY_OF_WEEK`, `HOUR_OF_DAY`, aggregated `INJURY` features)

### Feature Selection and Removal

| Name       | Action  | Reasoning                                                                                                                                                                 |
| ---------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `FATAL_NO` | Removed | This feature was identified as a sequential, auto-incrementing number assigned to _fatalities_, rather than a descriptive attribute of the collision or involved parties. |
| `ObjectID` | Removed | Removed as unique identifiers with no predictive value.                                                                                                                   |
| `Index`    | Removed | Removed as unique identifiers with no predictive value.                                                                                                                   |

### Addressing Data Granularity (if you aggregate to `ACCNUM` level)
