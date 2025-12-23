# Määrittele havainnoista kurssilla oppimasi perusteella seuraavat asiat ja esitä ne numeroina visualisoinnissasi:
# - Askelmäärä laskettuna suodatetusta kiihtyvyysdatasta
# - Askelmäärä laskettuna kiihtyvyysdatasta Fourier-analyysin perusteella
# - Keskinopeus (GPS-datasta)
# - Kuljettu matka (GPS-datasta)
# - Askelpituus (lasketun askelmäärän ja matkan perusteella)

# Esitä ainakin seuraavat kuvaajat:
# - Suodatettu kiihtyvyysdata, jota käytit askelmäärän määrittelemiseen.
# - Analyysiin valitun kiihtyvyysdatan komponentin tehospektritiheys
# - Reittisi kartalla

import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium


# ladataan data
df = pd.read_csv('./Data/Linear Acceleration.csv')



# low-pass filter
from scipy.signal import butter, filtfilt
def butter_lowpass_filter(data, cutoff, nyq, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y



# suodataan signaalia
signal = df['Linear Acceleration z (m/s^2)']
T_tot = df['Time (s)'].max()
n = len(df['Time (s)'])
fs = n / T_tot # näytteenottotaajuus

# suodattimen parametrit
nyq = fs/2
order = 3 
cutoff = 1/0.4 # cutoff taajuus (Hz)

# suodatettu signaali
filtered_signal = butter_lowpass_filter(signal, cutoff, nyq, order)



st.title('Fysiikan loppuprojekti')
# signaalin kuvaaja
fig, ax = plt.subplots(figsize=(12, 4))
plt.plot(df['Time (s)'], signal, label='signaali')
plt.plot(df['Time (s)'], filtered_signal, label='suodatettu signaali')
plt.title('Askelmittaus')
plt.ylabel('Kiihtyvyys Z')
plt.xlabel('Aika (s)')
plt.axis([0,420,-30,30])
plt.grid()
plt.legend()

st.title('Suodattaen kiihtyvyysdatan Z-komponentti')
st.pyplot(fig)



# signaalin zoomattu kuvaaja
fig_z, ax_z= plt.subplots(figsize=(12, 4))
plt.plot(df['Time (s)'], signal, label='signaali')
plt.plot(df['Time (s)'], filtered_signal, label='suodatettu signaali')
plt.title('Askelmittaus')
plt.ylabel('Kiihtyvyys Z')
plt.xlabel('Aika (s)')
plt.axis([30,60,-8,8])
plt.grid()
plt.legend()

st.title('Suodattaen kiihtyvyysdatan zoomattu Z-komponentti')
st.pyplot(fig_z)



# askeleiden laskeminen
steps1 = 0
for i in range(n-1):
    if filtered_signal[i]/filtered_signal[i+1] < 0 :
        steps1 = steps1 + 1/2 # Askelmäärä laskettuna suodatetusta kiihtyvyysdatasta

#print('Askelmäärä laskettuna suodatetusta kiihtyvyysdatasta: ', np.round(steps1))
st.write('Askelmäärä laskettuna suodatetusta kiihtyvyysdatasta: ', np.round(steps1))



# Askelmäärä laskettuna kiihtyvyysdatasta Fourier-analyysin perusteella
t = df['Time (s)'] # aika
N = len(signal) # näytteiden määrä
dt = np.max(t)/N # näytteenottoväli

#Fourier-analysis
fourier = np.fft.fft(signal,N) # Fourier-muunnos
psd = fourier*np.conj(fourier)/N # tehospektri
freq = np.fft.fftfreq(N,dt) # taajuudet
L = np.arange(1,int(N/2)) # poistetaan 0 ja negaativiset taajuudet

# Tehospektrin kuvaaja
fig_teshospektri, ax_tehospektri = plt.subplots(figsize=(12,4))
plt.plot(freq[L],psd[L].real)
plt.xlabel('Taajuus [Hz] = [1/s]')
plt.ylabel('Teho')

st.title('Tehospektri')
st.pyplot(fig_teshospektri)


f_max = freq[L][psd[L] == np.max(psd[L])][0] # dominantti taajuus
T = 1/f_max 
steps2 =  f_max*np.max(t) # Askelmäärä laskettuna kiihtyvyysdatasta Fourier-analyysin perusteella

print('Askelmäärä laskettuna kiihtyvyysdatasta Fourier-analyysin perusteella: ', np.round(steps2))
st.write('Askelmäärä laskettuna kiihtyvyysdatasta Fourier-analyysin perusteella: ', np.round(steps2))



# GPS data
df_location = pd.read_csv('./Data/Location.csv')

# poistetaan epätarkkaa dataa
df_location = df_location[df_location['Horizontal Accuracy (m)'] < 15]
df_location = df_location.reset_index(drop =True)

df_location['dist'] = np.zeros(len(df_location))
df_location['total_dist'] = np.zeros(len(df_location))
#df_location.head()


# Haversinen kaava
from math import radians, cos, sin, asin, sqrt
def haversine(lon1, lat1, lon2, lat2):
    # convert degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. 
    return c * r



# kahden pisteen välinen etäisyys
for i in range(len(df_location)-1):
    lon1 = df_location['Longitude (°)'][i]
    lon2 = df_location['Longitude (°)'][i+1]
    lat1 = df_location['Latitude (°)'][i]
    lat2 = df_location['Latitude (°)'][i+1]
    df_location.loc[i+1,'dist'] = haversine(lon1, lat1, lon2, lat2) * 1000 # in meters


# kokonaisetäisyys jokaisessa pisteessä
df_location['total_dist'] = df_location['dist'].cumsum()

# kokonaisetäisyys
total_distance = np.max(df_location['total_dist'])

# kokonaisaika
total_time = df_location['Time (s)'].max()

st.write('Askelpituus (lasketun askelmäärän ja matkan perusteella)', np.round(total_distance/steps2, 2), ' m')
# nopeus
average_speed_m_s = total_distance/total_time
average_speed_kph = average_speed_m_s * 3.6
st.write('Keskinopeus (GPS-datasta):', np.round(average_speed_kph, 2), ' km/h')
st.write('Kuljettu matka (GPS-datasta): ', np.round(total_distance, 2), ' m')





# kartan pohja
start_lat = df_location['Latitude (°)'].mean()
start_long = df_location['Longitude (°)'].mean()
my_map = folium.Map(location=[start_lat, start_long], zoom_start=16)

# Piirretään reitti kartalla
folium.PolyLine(df_location[['Latitude (°)', 'Longitude (°)']], color='blue', weight=3, opacity=1).add_to(my_map)

st.title('Reitti kartalla')
st_map = st_folium(my_map, width=900, height=650)