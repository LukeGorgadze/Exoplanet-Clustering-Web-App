import os
import streamlit as st
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

df = ""
planets = []

def norm2(vec):
    res = 0
    for el in vec:
        res += pow(el, 2)
    res = res ** (1. / 2)
    return res

class Planet:
    def __init__(self, name, moonNum, orbitalPeriod, planetRadius, planetMass, planetDensity, radialVelocity, ratioToStellar):
        self.name = name
        self.moonNum = moonNum
        self.orbitalPeriod = orbitalPeriod
        self.planetRadius = planetRadius
        self.planetMass = planetMass
        self.planetDensity = planetDensity
        self.radialVelocity = radialVelocity
        self.ratioToStellar = ratioToStellar
        self.position = np.array([moonNum, orbitalPeriod, planetRadius, planetMass, planetDensity, radialVelocity, ratioToStellar])
        self.cluster = 0
        self.color = "#FFFFFF"

    def setCluster(self, clusterIndex):
        self.cluster = clusterIndex

    def setColor(self, col):
        self.color = col

def fetchData():
    global df
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'DATA', 'ExoPlanets2.csv')
    df = pd.read_csv(data_path, sep=',')
    df.dropna(inplace=True)

def iterateThruData():
    for index in df.index:
        planet = Planet(
            df["pl_name"][index],
            df["sy_mnum"][index],
            df["pl_orbper"][index],
            df["pl_rade"][index],
            df["pl_bmasse"][index],
            df["pl_dens"][index],
            df["pl_rvamp"][index],
            df["pl_ratror"][index])
        global planets
        planets.append(planet)

def main():
    global doneSearching
    doneSearching = False

    st.title("Clustering Exoplanets with K-means Algorithm")
    st.write("This application clusters exoplanets using the K-means algorithm and visualizes the results.")

    fetchData()
    iterateThruData()

    st.header("Data Preview")
    st.dataframe(df.head())

    st.markdown("""I'm using a dataset from Nasa exoplanet science institute to gather information about planetary 
                systems that are outside of the solar system and with the help of K means 
                clustering I'm collecting similar planets in relevant groups. I believe my 
                project will help cosmic researchers to classify old and newly discovered 
                planets according to their parameters.
                """)
    
    st.markdown("""As you can see, number of clusters highly affect calculation time, after 50 clusters it might take up to 1 minute to run the program. I divided planets in groups based on 7 parameters, using k means, but plotted them with only 3 parameters. My pseudo visualisation gives same color to similar planets, and lastly, My clustering algorithm will always converge if number of Ks is between 1-600.
                You can add data to my csv file and I will be able to use it""")
    
    st.subheader("After you select the number of clusters, the program will start to run. It will take some time to run, please be patient.")

    K = st.slider("Choose Number of Clusters (K)", min_value=2, max_value=100, step=2, value=5)
    Ks = []
    for i in range(K):
        randy = random.randint(0, 600)
        Ks.append(planets[randy].position + (random.random() * .5 - 1) * 200)

    clusters = {}

    def reGroup():
        for i in range(K):
            clusters.update({i: []})
        for pIndex, planet in enumerate(planets):
            minDist = 999999999
            clusterNum = 0
            for kIndex, kPoint in enumerate(Ks):
                dist = norm2(planet.position - kPoint)
                if (dist < minDist):
                    minDist = dist
                    clusterNum = kIndex
            planet.setCluster(clusterNum)
            dic = clusters[clusterNum]
            dic.append(planet)
            clusters.update({clusterNum: dic})

    def updateKs():
        count = 0
        tempKs = Ks
        for index, kPoint in enumerate(tempKs):
            newVec = [0] * 7
            count = len(clusters[index])
            for planet in clusters[index]:
                for ind, v in enumerate(planet.position):
                    if count > 0:
                        newVec[ind] += v / count
            Ks[index] = newVec
        return Ks

    iteration = 0
    prevks = []
    reGroup()
    currKs = updateKs()

    t = st.empty()
    while not doneSearching:
        currKs = updateKs()
        doneSearching = currKs == prevks
        reGroup()
        iteration += 1
        prevks = currKs.copy()
        t.write(f'Loading... Iteration: {iteration} 😳')

    t.empty()
    # st.write("Here are the results 😎😎😎:")
    # for key, value in clusters.items():
    #     st.write(f"Cluster {key}:")
    #     for p in value:
    #         st.write(f"Name: {p.name}, Density: {p.planetDensity}, RatioToStellar: {p.ratioToStellar}, Planet Mass: {p.planetMass}")

    # Assign colors to clusters
    for key, value in clusters.items():
        hexV = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'A', 'B', 'C', 'D', 'E', 'F']
        col = ""
        for i in range(6):
            col += str(hexV[random.randint(0, len(hexV) - 1)])
        for p in value:
            p.setColor("#" + col)

    # Visualization using Matplotlib
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    random.seed(10)
    for p in planets:
        x = p.planetDensity + random.random() * 100
        y = p.ratioToStellar + random.random() * 100
        z = p.planetMass + random.random() * 5000
        color = p.color
        radius = p.planetRadius * 2
        ax.scatter(x, y, z, c=color, s=radius, alpha=0.7)

    ax.set_xlabel("Planet Density")
    ax.set_ylabel("Ratio to Stellar")
    ax.set_zlabel("Planet Mass")
    ax.set_title("Clustering of Exoplanets")

    st.header("3D Visualization of Exoplanets Clustering")
    st.pyplot(fig)

    # Results Table
    st.header("Results Table 😎😎😎")
    
    result_data = {'Cluster': [], 'Name': [], 'Density': [], 'RatioToStellar': [], 'Planet Mass': []}
    
    for key, value in clusters.items():
        for p in value:
            result_data['Cluster'].append(f"Cluster {key}")
            result_data['Name'].append(p.name)
            result_data['Density'].append(p.planetDensity)
            result_data['RatioToStellar'].append(p.ratioToStellar)
            result_data['Planet Mass'].append(p.planetMass)
    
    result_df = pd.DataFrame(result_data)
    st.table(result_df)

if __name__ == "__main__":
    main()
