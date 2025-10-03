import pandas as pd
stupci = ['Speed over Ground [knots]', 'Heading [degrees]', 'Shaft RPM PS [rpm]', 'Shaft RPM SB [rpm]',
          'Shaft Power PS [kW]', 'Shaft Power SB [kW]', 'Shaft Torque PS [kNm]', 'Shaft Torque SB [kNm]',
          'Wind Speed [m/s]', 'Consumption 10 minutes ago', 'Consumption 20 minutes ago',
          'Consumption 30 minutes ago', 'Consumption 40 minutes ago', 'Consumption 50 minutes ago']

# ti podaci su iz vremena: od 2025-08-30 21:30:00  do 2025-08-30 23:00:00   

podaci = [
    [18.9, 267.5, 213.0	, 213.2, 3544, 3581, 308.3, 313, 6.5, 2150.8, 2154.9, 2155.7, 2154.3, 2156.3],
    [17.8, 190.9, 213.7, 212.8, 3585, 3567, 310.5, 311.6, 9.6, 2154.9, 2150.8, 2154.9, 2155.7, 2154.3],
    [17.2,100,213.2,212.8,3522,3554,306.9,310.5,8.7,2173.9,2154.9,2150.8,2154.9,2155.7],
    [17.3,88.3,213.2,212.8,3594,3481,313.4,303.6,14.4,2156.4,2173.9,2154.9,2150.8,2154.9],
    [17.3,89,212.6,213.5,3431,3499,298.5,305,7.5,2151.7,2156.4,2173.9,2154.9,2150.8],
    [12.9,89.4,163.3,163.1,1602,1552,187.1,181.3,7.5,2143.6,2151.7,2156.4,2173.9,2154.9],
    [13,89.8,163.1,163.3,1583,1561,184.5,182.4,7.7,1106.3,2143.6,2151.7,2156.4,2173.9],
    [13,90.3,163.1,163.3,1592,1533,186.4,179.8,8.9,1086.2,1106.3,2143.6,2151.7,2156.4],
    [13.1,89.9,163.3,163.1,1579,1561,184.2,183.1,9.7,1089.1,1086.2,1106.3,2143.6,2151.7],
    [13.5,89.3,163.3,163.3,1597,1542,185.6,180.9,10.1,1101.6,1089.1,1086.2,1106.3,2143.6]
]

df = pd.DataFrame(podaci, columns=stupci)
df.to_csv(r"C:/Users/BrunoNad/Documents/Project_consumption/test_fuel_consumption_data.csv", index=False)

df.to_excel("C:/Users/BrunoNad/Documents/Project_consumption/test_data.xlsx")



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