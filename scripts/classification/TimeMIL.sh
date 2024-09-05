model_name=TimeMIL

## Paper experiment list
for dataset in ArticularyWordRecognition AtrialFibrillation BasicMotions Cricket DuckDuckGeese EigenWorms Epilepsy EthanolConcentration ERing FaceDetection FingerMovements HandMovementDirection Handwriting Heartbeat Libras LSST MotorImagery NATOPS PenDigits PEMS-SF PhonemeSpectra RacketSports SelfRegulationSCP1 SelfRegulationSCP2 StandWalkJump UWaveGestureLibrary
do 
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/$dataset/ \
  --root_path ./dataset/$dataset/ \
    --model_id $dataset \
  --model $model_name \
  --data UEA \
  --learning_rate 0.001 \
  --train_epochs 200 \
  --batch_size 64 \
  --patience 20 \
  --d_model 128  \
  --dropout 0.2 \
  --use_gpu True \
  --num_workers 0 \
  --epoch_des 10 \
  --dropout_patch 0.5 \
  --weight_decay 0.0001
done

for dataset in EthanolConcentration FaceDetection Handwriting Heartbeat JapaneseVowels PEMS-SF SelfRegulationSCP1 SelfRegulationSCP2 SpokenArabicDigits UWaveGestureLibrary
do 
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/$dataset/ \
  --root_path ./dataset/$dataset/ \
    --model_id $dataset \
  --model $model_name \
  --data UEA \
  --learning_rate 0.001 \
  --train_epochs 200 \
  --batch_size 64 \
  --patience 20 \
  --d_model 128  \
  --dropout 0.2 \
  --use_gpu True \
  --num_workers 0 \
  --epoch_des 10 \
  --dropout_patch 0.5 \
  --weight_decay 0.0001
done

# #Univariate
# for dataset in Haptics SyntheticControl Worms Computers HouseTwenty GestureMidAirD3 Chinatown UWaveGestureLibraryAll Strawberry Car GunPointAgeSpan GestureMidAirD2 BeetleFly Wafer CBF Adiac ItalyPowerDemand Yoga AllGestureWiimoteY  Trace PigAirwayPressure ShapesAll Beef GesturePebbleZ2 Mallat GunPointOldVersusYoung MiddlePhalanxTW AllGestureWiimoteX Meat Herring MiddlePhalanxOutlineCorrect InsectEPGRegularTrain FordA SwedishLeaf InlineSkate DodgerLoopDay UMD CricketY WormsTwoClass SmoothSubspace OSULeaf Ham CricketX SonyAIBORobotSurface1 ToeSegmentation1 ScreenType PigArtPressure SmallKitchenAppliances Crop MoteStrain MelbournePedestrian ECGFiveDays Wine SemgHandMovementCh2 FreezerSmallTrain UWaveGestureLibraryZ NonInvasiveFetalECGThorax1 TwoLeadECG Lightning7 Phoneme SemgHandSubjectCh2 DodgerLoopWeekend MiddlePhalanxOutlineAgeGroup GestureMidAirD1 DistalPhalanxOutlineCorrect DistalPhalanxTW FacesUCR ECG5000 ShakeGestureWiimoteZ GesturePebbleZ1 HandOutlines GunPointMaleVersusFemale Coffee Rock MixedShapesSmallTrain AllGestureWiimoteZ FordB FiftyWords InsectWingbeatSound MedicalImages Symbols ArrowHead ProximalPhalanxOutlineAgeGroup EOGHorizontalSignal TwoPatterns ChlorineConcentration Plane ACSF1 PhalangesOutlinesCorrect ShapeletSim DistalPhalanxOutlineAgeGroup InsectEPGSmallTrain PickupGestureWiimoteZ EOGVerticalSignal CricketZ FaceFour RefrigerationDevices PLAID MixedShapesRegularTrain GunPoint DodgerLoopGame ECG200 ToeSegmentation2 WordSynonyms Fungi BirdChicken SemgHandGenderCh2 OliveOil BME LargeKitchenAppliances SonyAIBORobotSurface2 Lightning2 EthanolLevel UWaveGestureLibraryX FreezerRegularTrain Fish ProximalPhalanxOutlineCorrect NonInvasiveFetalECGThorax2 UWaveGestureLibraryY FaceAll StarLightCurves ElectricDevices Earthquakes PowerCons DiatomSizeReduction CinCECGTorso PigCVP ProximalPhalanxTW
# do
# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/$dataset/ \
#   --root_path ./dataset/$dataset/ \
#     --model_id $dataset \
#   --model $model_name \
#   --data UEA \
#   --learning_rate 0.001 \
#   --train_epochs 200 \
#   --batch_size 64 \
#   --patience 200 \
#   --d_model 128  \
#   --dropout 0.2 \
#   --use_gpu False \
#   --num_workers 0 \
#   --epoch_des 10 \
#   --dropout_patch 0.5 \
#   --weight_decay 0.0001
# done


# for dataset in 'LSST SelfRegulationSCP2 FaceDetection MotorImagery PenDigits Libras PhonemeSpectra InsectWingbeat Cricket Handwriting ArticularyWordRecognition StandWalkJump CharacterTrajectories ERing HandMovementDirection SelfRegulationSCP1 JapaneseVowels Heartbeat RacketSports EigenWorms FingerMovements PEMS-SF Epilepsy NATOPS AtrialFibrillation SpokenArabicDigits EthanolConcentration BasicMotions DuckDuckGeese UWaveGestureLibrary'
# do
# python -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/$dataset/ \
#   --root_path ./dataset/$dataset/ \
#     --model_id $dataset \
#   --model $model_name \
#   --data UEA \
#   --learning_rate 0.001 \
#   --train_epochs 200 \
#   --batch_size 64 \
#   --patience 200 \
#   --d_model 128  \
#   --dropout 0.2 \
#   --use_gpu False \
#   --num_workers 0 \
#   --epoch_des 10 \
#   --dropout_patch 0.5 \
#   --weight_decay 0.0001
# done
