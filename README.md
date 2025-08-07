# Clear Sky Day Filter for Photovoltaic Monitoring Data
A Clear Sky Day filter, that takes monitoring data of PV systems as power over time and returns only the data of clear sky days.

<img width="3065" height="2302" alt="ClearSkyDay-GraphicalAbstract" src="https://github.com/user-attachments/assets/0156f08b-f373-40a0-b150-21ce0f698af8" />
Figure 1: The Clear Sky filter takes several days of monitoring data and returns only days, classified as clear sky days.

If you use the code in terms of a publication, I would be greatfull, if you could cite the following paper:
E. Wittmann, C. Buerhop-Lutz, S. Bennett, V. Christlein, J. Hauch, C. J. Brabec, I. M. Peters, „PV Polaris – Automated PV system Orientation Prediction”, IEEE Photonics Journal, vol. 17, no. 3, 2025. DOI: 10.1109/JPHOT.2025.3568887
in which the clear sky filter is published.

##Functionality:
First a clear sky template is created based on the work of Ian Marius et all. [1]. The process involves finding the maximum power output for each time of day over a one-month dataset, multiplying the result by a percentile (e.g., 0.9 for Germany), and smoothing it using median and mean sliding windows.
The clear sky filter than filters out days, that show lagging data. To filter days with lagging data, there are three checks:
1. Check if data over a day has enough datapoints. (e.g. within a recordingspeed of 1h, there should be 24 datapoints for a day)
2. Check if there are data holes.
3. Check if the power curve over a day starts and ends by low power. (day starts and ends at night and during night the power generation is typically low)
Afterwards the remaining days are compared with the template by Euclidian distance and correlation thresholds for similarity. If a day is similar (low distances and high correlation), the day is defined as clear sky day.

## Input Data:
As input each kind of monitoring data - module, string or inverter level - is possible. The monitoring data has to include the power over time.
The data has to be given to the algorithm as a polars dataframe with the time as "date_time" and the power as float32 or float64:

import polars as pl
data = pl.read_csv("data.csv")
data = data.with_columns(pl.col("time").str.to_datetime(), pl.col("power").cast(pl.Float64))

 
 If there are more than one PV system included in the date, (e.g. the power is recorded by several strings) the data can be stored in two different kinds (see Figure 2).
 Both ways are fine.
 <img width="3667" height="977" alt="DataExample" src="https://github.com/user-attachments/assets/28b30ff9-ab82-48b6-9597-1fadf87cae8e" />
Figure 2: Example of monitoring data including several PV systems. The data can be stored either with all power in one power column (left) and an additional "id"-column or with a power column for each PV system (right).

If the data looks like Figure 2 - Left, than the three variables: "column_time", "column_power", "column_id" have to be set. 
Column_time contains the name of the column including the times as datetime.
Column_power contains the name of the column including the power as Float.
Column_id contains the PV system ids.
The code will automatically go through every id and return for each id, the clear sky days.

If the data looks like Figure 2 - Right, than only the two varaibles: "column_time", and "column_power" have to be set.
Column_time contains the name of the column including the times as datetime.
Column_power contains the name of the one column including the power as Float.
The code will only search for clear sky days of this one PV system. If the clear sky days have to be search for another PV system, than the algorithm has to be used another time, with another power column.

## Template:
To create the template, that is used for clear sky comparison, a comparison_intervall over several days has to be set. The template algorithm will overlap all days over each other, and for each time step over a day, will take the maximal value. The default comparison_intavall is 30 days. It is recommended to have at least 5 days. Before the data is processed, the days can be smoothed by a 
mean sliding window with keneral size "prep_smooth_kernal". The idea behind this kind of smoothing is, to reduce strong cloud noises in the data. After the maximation process, another smoothing can be done with a mean sliding window by given kernal size "smooth_kernal". To deactivate the smoothings, both can be set to "None". Expirience has shown, that best results can be achieved by having a small prep_smoothing_kernal (2-10 by a recording speed of 1min) and a large smoothing_kernal (40-80 by a recording speed of 1min).  In the end there can be a percentil be used. The percentil is a correction factor since often the noise is leading to higher power values (due to e.g. cloud lensing effect, or air polution) than actuall given. In Germany the percentil is aroung 0.9 (see Figure 3).

<img width="4297" height="1810" alt="Template_Params" src="https://github.com/user-attachments/assets/3b815332-7869-45a2-84a0-32180b4c8716" />
Figure 3: Impact of smoothing on the template. The example data includes 60 days with a recording speed of 1 min. No smoothing leads to a noisy template. Using prep_smoothing, the template shows less noise. However, the power dorps down. Using only the smoothing leads to a template with low noise. Using both kernals lead to the best clear sky similar result.  

 
 

