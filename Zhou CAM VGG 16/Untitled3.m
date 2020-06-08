x = 1:5;
Mammography = [80.5, 73.3, 85.4, 66.0, 76.9];
Thermography = [81.6, 57.8, 78.9, 61.9, 69.7];
ThermoIceTest = [92.8, 41.9, 75.5, 75, 67.35];
MammoAndThermo = [96.2, 44.4, 77.1, 87, 70.5];
My200Times = [100, 86.15, 25, 100, 93.08];
MyBalancedClass = [100, 60, 33.33, 100, 80];

%Sensitivity, Specificity, PPV, NPV, Accuracy
% plot(x, Mammography, 'r:+', x, Thermography, 'r:o', x, ThermoIceTest, 'r:*', x, MammoAndThermo, 'r:x');
whitebg ([1 1 1]);

plot(x, Mammography, '-+c', x, Thermography, '-oy', x, ThermoIceTest, '-*r',...
    x, MammoAndThermo, '-xg', x, My200Times, '-sb', x, MyBalancedClass, '-^k');


legend('Mammography','Thermography','Thermo + Ice Test', 'Mammo and Thermo', 'DataIncremented', 'Balanced Class')
