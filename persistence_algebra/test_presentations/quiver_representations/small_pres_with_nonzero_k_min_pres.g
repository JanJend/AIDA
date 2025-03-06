Q := Quiver(3, [ [1,3, "E0"], [2,3, "E1"]]);
AQ := PathAlgebra(GF(2), Q);
MQ := RightModuleOverPathAlgebra(AQ, [2,1,1], [ ["E0", [[Z(2)^0],[Z(2)^0]] ], ["E1", [[Z(2)^0]] ] ]);
