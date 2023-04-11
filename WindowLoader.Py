import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import math

# takes in as arguments which teams the data will be read from and also the amount of time in seconds that each
# reshaped sample will span in the original time series data
class DataReader(Dataset):
    def __init__(self, window_size, teams = []):
        self.teams = teams
        self.sec_per_frame=0.008333

        # calculate number of rows that will be flattened into a single row to form each sample
        self.frames_per_window = math.floor(window_size/self.sec_per_frame)
        self.X_tens, self.Y_tens = self.TensorCreator()


    def TensorCreator(self):
        # read csv file and load row data into variables
        # Labels are: 0 STANDING, 1 SITTING, 2 WALKING
        data = []


        # Initial data storage arrays must have 6 * frames_per_window elements. The 6 features per roww are
        # x,y,z rotations, y position, vertical & horizontal velocity
        X_tens = np.ndarray([0, 6 * self.frames_per_window])
        Y_tens = []

        # iterate through teams
        for team in self.teams:
            # iterate through users
            for i in range(1,3):
                for j in range(1,19):
                    # Handle missing data: team 1 does not have a user 2 and user 3 does not have data past take/file 12
                    if (team == 1 and i == 2) or (team == 1 and j == 13):
                        break

                    # create and modify file name string
                    file_name = 'data/Team'+str(team)+'/t' + str(team) + '_User_0' + str(i) + \
                               ('_0' if j >= 10 else '_00') + str(j) + '.csv'

                    # read in file csv as dataframe
                    file_out = pd.read_csv(file_name,header=0,skiprows=6)

                    # cut and label data: modulus chooses label, j value chooses where data is located in dataframe
                    if (j + 2) % 3 == 0:
                    # STANDING 0
                        if j <= 9:
                            data = [file_out.iloc[0:, 2:8], 0]
                        elif j > 9 and j < 19:
                            data = [file_out.iloc[0:, 29:35], 0]
                    elif (j+1)%3==0:
                    # SITTING 1
                        if j <= 9:
                            data = [file_out.iloc[0:, 2:8], 1]
                        elif j > 9 and j < 19:
                            data = [file_out.iloc[0:, 29:35], 1]

                    elif j % 3 == 0:
                    # WALKING 2
                        if j <= 9:
                            data = [file_out.iloc[0:, 2:8], 2]
                        elif j > 9 and j < 19:
                            data = [file_out.iloc[0:, 29:35], 2]

                    # find index of the last non-NAN valued row in data and keeps it as valid length
                    # all NAN values beneath this index will be filled in via linear interpolation.
                    switch = np.argwhere(~np.isnan(data[0].values))
                    nonNAN_len = switch[len(switch) - 1][0] + 1

                    # axis '0' is columns, so fills in missing vals via interpolation going downwards
                    data[0] = data[0].interpolate(method='linear', axis=0)
                    # stores data up to last non-NAN interpolated row index
                    data[0] = data[0].values[0:nonNAN_len, 0:]

                    # intialize array to store vertical and horizontal velocities
                    tempData = np.ndarray([0, 6])

                    for k in range(1, len(data[0])):
                        # compute vertical velocity along y direction, use index 4 corresponding to
                        # [x rot,y rot,z rot,x pos,y pos,z pos]
                        vert_velocity = (data[0][k, 4] - data[0][k - 1, 4]) / self.sec_per_frame

                        # compute horizontal velocity along x & z directions. Use index 3 for x and index 5 for z
                        horiz_velocity = math.sqrt(
                            (data[0][k, 3] - data[0][k - 1, 3]) ** 2 + (data[0][k, 5] - data[0][k - 1, 5]) ** 2) \
                                         / self.sec_per_frame

                        # make first 3 entries the x,y,z rotations, and the next entries the y (index 4), and
                        # vertical/horizontal velocities
                        row = np.hstack((data[0][k, 0:3], np.array([data[0][k, 4], vert_velocity, horiz_velocity])))
                        tempData = np.vstack([tempData, row])

                    # cut data so that it may be evenly reshaped to have rows with no missing values
                    valid_length = len(tempData) - len(tempData) % self.frames_per_window
                    tempData = tempData[0:valid_length, 0:]

                    # reshape data to form samples corresponding to chosen window size
                    data[0] = np.reshape(tempData, (-1, 6 * self.frames_per_window), 'C')

                    # store data and labels
                    X_tens = np.vstack([X_tens,data[0]])
                    Y_tens += [data[1] for x in range(data[0].shape[0])]

        # wrap data and labels in tensors
        X_tens = torch.tensor(X_tens, dtype=torch.float32)
        Y_tens = torch.tensor(Y_tens)

        return (X_tens,Y_tens)

    def __len__(self):
        return len(self.Y_tens)

    def __getitem__(self,idx):
        return self.X_tens[idx], self.Y_tens[idx]

