160404
09:54
knut.mora@fysik.su.se

The two files ClassSample_training_0.npy and ClassSample_training_1.npy contain real OFF data and gamma MC respectively
taken with a Cherenkov Telescope. 

each file may be loaded with np.load

Each file contains 9 colums, where the last one signifies background =0 or signal =1: 

Column 0: Theta2 [deg^2] is the distance from the (fiducial) source position and the reconstructed event
Column 1: MeanScaledShowerGoodness [1] is a goodness of fit variable output from the reconstruction assuming that the
event is a photon shower
Column 2: Direction Error [1] is an estimated error output from the reconstruction
Column 3: Primary Depth measures how far into the atmosphere the particle first interacted
Column 4: MeanScaledBackgroundGoodness is a GOF assuming background
Column 5: NSBLikelihood is a ~GOF assuming the event is Night Sky Background- starlight etc. 
Column 6: LogEnergy [1] is log10(Erec/1TeV) the reconstructed energy
Column 7: Core measures hod far into the atmosphere the shower maximum is
Column 8: is 0 if background, 1 if gamma
