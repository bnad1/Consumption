import pandas as pd
stupci = ['Speed over Ground [knots]', 'Heading [degrees]', 'Shaft RPM PS [rpm]', 'Shaft RPM SB [rpm]',
          'Shaft Power PS [kW]', 'Shaft Power SB [kW]', 'Shaft Torque PS [kNm]', 'Shaft Torque SB [kNm]',
          'Wind Speed [m/s]', 'Consumption 10 minutes ago', 'Consumption 20 minutes ago',
          'Consumption 30 minutes ago', 'Consumption 40 minutes ago', 'Consumption 50 minutes ago']

podaci = [
    [18.9, 267.5, 213.0	, 213.2, 3544, 3581, 308.3, 313, 6.5, 2150.8, 2154.9, 2155.7, 2154.3, 2156.3],
    [17.8, 190.9, 213.7, 212.8, 3585, 3567, 310.5, 311.6, 9.6, 2154.9, 2150.8, 2154.9, 2155.7, 2154.3],
    [17.2,100,213.2,212.8,3522,3554,306.9,310.5,8.7,2173.9,2154.9,2150.8,2154.9,2155.7]
]

df = pd.DataFrame(podaci, columns=stupci)
df.to_csv(r"C:/Users/BrunoNad/Documents/Project_consumption/test_fuel_consumption_data.csv", index=False)

stupci1=['Speed over Ground [knots]','Heading [degrees]','Shaft RPM PS [rpm]','Shaft RPM SB [rpm]','Shaft Power PS [kW]','Shaft Power SB [kW]','Shaft Torque PS [kNm]',
         'Shaft Torque SB [kNm]','Wind Speed [m/s]','Consumption 5 minutes ago','Consumption 10 minutes ago','Consumption 15 minutes ago','Consumption 20 minutes ago',
         'Consumption 25 minutes ago','Consumption 30 minutes ago','Consumption 35 minutes ago','Consumption 40 minutes ago','Consumption 45 minutes ago',
         'Consumption 50 minutes ago','Consumption 55 minutes ago','Consumption 60 minutes ago']

podaci1=[
  [18.975,267.55,212.9,213.2,3539,3528.75,308.225,307.950,7,2150.8,2155.8,2154.9,2149.4,2155.7,2155.5,2154.3,2149.3,2156.3,2155.4,2152.6,2172],
  [18.9,267.5,213,213.2,3544,3581,308.3,313,6.5,2145.7,2150.8,2155.8,2154.9,2149.4,2155.7,2155.5,2154.3,2149.3,2156.3,2155.4,2152.6],
  [18.5,236.5,213,213,3541,3571.75,308.5,312.475,7.8,2154.9,2145.7,2150.8,2155.8,2154.9,2149.4,2155.7,2155.5,2154.3,2149.3,2156.3,2155.4],
]

df1=pd.DataFrame(podaci1,columns=stupci1)
df1.to_csv(r"C:/Users/BrunoNad/Documents/Project_consumption/test1_fuel_consumption_data.csv", index=False)
         

stupci2 = ['Speed over Ground [knots]', 'Heading [degrees]', 'Shaft RPM PS [rpm]',
          'Shaft RPM SB [rpm]', 'Shaft Power PS [kW]', 'Shaft Power SB [kW]', 
          'Shaft Torque PS [kNm]', 'Shaft Torque SB [kNm]', 'Rate of turn','Wind Speed [m/s]',
          'Consumption 10 minutes ago', 'Consumption 20 minutes ago',
          'Consumption 30 minutes ago', 'Consumption 40 minutes ago',
          'Consumption 50 minutes ago']

podaci2=[
  [18.9,267.5,213,213.2,3544,3581,308.3,313,0.04,6.5,2150.8,2154.9,2155.7,2154.3,2156.3],
  [17.8,190.9,213.7,212.8,3585,3567,310.5,311.6,-7.66,9.6,2154.9,2150.8,2154.9,2155.7,2154.3],
  [17.2,100,213.2,212.8,3522,3554,306.9,310.5,-9.09,8.7,2173.9,2154.9,2150.8,2154.9,2155.7],
]

df2=pd.DataFrame(podaci2,columns=stupci2)
df2.to_csv(r"C:/Users/BrunoNad/Documents/Project_consumption/test2_fuel_consumption_data.csv", index=False)