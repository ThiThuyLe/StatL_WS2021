import random as rd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
#=============================================================================================================
Episodenanzahl = 300  # Anzahl der Durchläufe
max_Schritte = 100  # Abbruchparameter
epsilon_max = 1  # Starterkundungsrate
epsilon_red = 0.01  # Erkundungsreduktion
epsilon_min = 0.001  # Minimale Erkundungsrate
fressen = 1  # Belohnung
kollision = -1  # Bestrafung
laufen = 0.1  # Belohnung
gamma = 0.9  # Diskontierungsfaktor
alpha = 0.5  # Lernrate
Feld_Reihen = 10
Feld_Spalten = 10
Erinnerungsspeicher = 700
Erinnerungsset = 70
DQL_akt_rate = 10
hidden_layer_1 = 200
Hidden_layer_2 = 160
Hidden_layer_3 = 100
#=============================================================================================================
Belohnungsvektor = []
Schrittzähler = []
#=============================================================================================================
Aktionen = (
    [ 0, 1],  # rechts = 0
    [-1, 0],  # oben   = 1
    [ 0,-1],  # links  = 2
    [ 1, 0]  # unten   = 3
)
#=============================================================================================================
class Netzwerk(nn.Module):
    def __init__(self):
        super().__init__()
        self.f1 = nn.Linear(6, hidden_layer_1)
        self.f2 = nn.Linear(hidden_layer_1, Hidden_layer_2)
        self.f3 = nn.Linear(Hidden_layer_2, Hidden_layer_3)
        self.f4 = nn.Linear(Hidden_layer_3, 4)
        
    def forward(self, t):
        t = self.f1(t)
        t = F.relu(t)
        t = self.f2(t)
        t = F.relu(t)
        t = self.f3(t)
        t = F.relu(t)
        t = self.f4(t)
        return t

class Snake(object):
    def __init__(self):
        self.epsilon = epsilon_max
        self.Erinnerungen = []
        self.Episodennummer = 1
        self.Reihen = Feld_Reihen
        self.Spalten = Feld_Spalten
        self.Laenge = Feld_Reihen//4
        self.Schlange_Start_Position = \
            [[Feld_Reihen//2 + self.Laenge - r - 1, Feld_Spalten//2]
                for r in range(self.Laenge)]
        self.Belohnung = 0
        self.game_over = False

    def get_Freiflaeche(self):
        self.Freiflaeche = \
            [[r, s] for r in range(self.Reihen) for s in range(self.Spalten) if [r, s] not in self.Schlange_Position]
        return self.Freiflaeche

    def Abbruch(self):
        if self.Schrittanzahl >= max_Schritte or self.game_over:
            return True
        else:
            return False

    def Neustart(self):
        self.Schlange_Position = self.Schlange_Start_Position.copy()
        self.Apfel_erzeugen()
        self.Belohnung_Gesamt = 0
        self.game_over = False
        self.Schrittanzahl = 0
        self.get_Zustand()
        self.Laenge = len(self.Schlange_Position)

    def get_Zustand(self):
        oben = unten = links = rechts = 0 # Abstand zur nächsten Gefahr
        self.get_Freiflaeche()
        while list(np.add(self.Schlange_Position.copy()[-1], [0, rechts+1])) in self.Freiflaeche:
            rechts += 1
        while list(np.add(self.Schlange_Position.copy()[-1], [oben-1, 0])) in self.Freiflaeche:
            oben += 1
        while list(np.add(self.Schlange_Position.copy()[-1], [0, links-1])) in self.Freiflaeche:
            links += 1
        while list(np.add(self.Schlange_Position.copy()[-1], [unten+1, 0])) in self.Freiflaeche:
            unten += 1
        self.Zustand = [rechts,
                        oben,
                        links,
                        unten,
                        self.Apfel_Position[0] - self.Schlange_Position[-1][0],
                        self.Apfel_Position[1] - self.Schlange_Position[-1][1]]
        return self.Zustand

    def Apfel_erzeugen(self):
        Valide_position = False
        trys = 0
        while not Valide_position and trys < 40:
            Apfel_Reihe = rd.randint(0, self.Reihen - 1)
            Apfel_Spalte = rd.randint(0, self.Reihen - 1)
            if list([Apfel_Reihe, Apfel_Spalte]) not in self.Schlange_Position:
                self.Apfel_Position = [Apfel_Reihe, Apfel_Spalte]
                Valide_position = True
            else:
                trys += 1
        if trys == 40:
            print('to many trys')
            self.game_over = True
            self.Apfel_Position = [0,0]

    def Bewegung(self):
        self.Zustand_alt = self.Zustand.copy()
        self.get_Freiflaeche()
        self.Schlange_Position.append(list(np.add(self.Schlange_Position[self.Laenge-1], self.Aktion)))
        self.Schrittanzahl += 1
        if self.Schlange_Position[self.Laenge] in self.Freiflaeche:
            if not self.Apfel_Position == self.Schlange_Position[self.Laenge]:
                del self.Schlange_Position[0]
                self.Belohnung = laufen
            else:
                self.Apfel_erzeugen()
                self.Belohnung = fressen
            self.Laenge = len(self.Schlange_Position)
        else:
            self.game_over = True
            self.Belohnung = kollision
            self.Laenge = len(self.Schlange_Position)
        self.get_Zustand()

    def Plot_Dokumentation(self):
        Belohnungsvektor.append([self.Episodennummer, self.Belohnung_Gesamt])
        Schrittzähler.append([self.Episodennummer, self.Schrittanzahl])

    def Erinnerung_abspeichern(self):
        self.Erinnerungen.append([self.Zustand_alt, self.Richtung, self.Belohnung, self.Zustand])
        if len(self.Erinnerungen) > Erinnerungsspeicher:
            del self.Erinnerungen[0]   

    def Exploration_vs_Exploitation(self, NW):
        if rd.random() < self.epsilon:
            self.Richtung = rd.choice(range(len(Aktionen)))
        else:
            self.Richtung = torch.argmax(NW(torch.FloatTensor(self.Zustand)))
        self.Aktion = Aktionen[self.Richtung]
        
    def Netz_trainieren(self, PNW, TNW):
        if len(self.Erinnerungen) <= Erinnerungsset:
            self.trainings_set = self.Erinnerungen
        else:
            self.trainings_set = rd.sample(self.Erinnerungen, Erinnerungsset)
        q = []
        q_stern = []
        for Erinnerung in self.trainings_set:   
            q.append(PNW(torch.FloatTensor(Erinnerung[0]))[Erinnerung[1]])
            clipped_min = torch.minimum(
                Erinnerung[2] + gamma * torch.argmax(TNW(torch.FloatTensor(Erinnerung[3]))),
                Erinnerung[2] + gamma * torch.argmax(PNW(torch.FloatTensor(Erinnerung[3])))
                )
            q_stern.append(clipped_min)
        return torch.FloatTensor(q_stern).requires_grad_(True), torch.FloatTensor(q).requires_grad_(True)

    def Erkundung_verringern(self):
        self.epsilon = epsilon_min + (epsilon_max - epsilon_min) / np.exp(epsilon_red * self.Episodennummer)
            
Spiel = Snake()
Strategie_NW = Netzwerk()
Target_NW = Netzwerk()
Target_NW.load_state_dict(Strategie_NW.state_dict())
Target_NW.eval()
optimierer = torch.optim.Adam(Strategie_NW.parameters(), lr=alpha)
lossliste = []
while Spiel.Episodennummer <= Episodenanzahl:
    Spiel.Neustart()
    while not Spiel.Abbruch():
        Spiel.Exploration_vs_Exploitation(NW=Strategie_NW)
        Spiel.Bewegung()
        Spiel.Erinnerung_abspeichern()
        Spiel.Belohnung_Gesamt += Spiel.Belohnung
        q_stern, q = Spiel.Netz_trainieren(PNW=Strategie_NW, TNW=Target_NW)
        loss = F.mse_loss(q_stern, q)
        optimierer.zero_grad()
        loss.backward()
        optimierer.step()
    Spiel.Erkundung_verringern()
    Spiel.Plot_Dokumentation()
    lossliste.append([Spiel.Episodennummer, loss.detach().numpy()])
    if Spiel.Episodennummer % DQL_akt_rate == 0:
        Target_NW.load_state_dict(Strategie_NW.state_dict())
        Target_NW.eval()
    Spiel.Episodennummer += 1


plt.plot([row[0] for row in Schrittzähler], [row[1]
         for row in Schrittzähler], 'r', label='Anzahl der Schritte')
plt.plot([row[0] for row in lossliste], [row[1]
         for row in lossliste], 'b', label='Loss')
plt.plot([row[0] for row in Belohnungsvektor], [row[1]
         for row in Belohnungsvektor], 'g', label='Höhe der Belohnung')
plt.xlabel('Episoden')
plt.legend()
plt.show()