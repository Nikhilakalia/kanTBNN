#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 17:25:35 2021

@author: ryley
"""
import numpy as np

def writeFoam_U_DUCT(filename, U):
    # Writes the aperp file
    cells = len(U)
    leftbracket = np.repeat('(',cells)
    rightbracket = np.repeat(')',cells)
    array_write = np.column_stack((leftbracket,U,rightbracket))
    print('[dataFoam] Writing U to file '+filename)
    file = open(filename,'w')
    file.write("""/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2006                                 |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];

internalField nonuniform List<vector>
""")
    file.write(str(cells) + """ (
""")
    np.savetxt(file, array_write,fmt='%s')
    file.write(""");
boundaryField
{
    inlet
    {
        type            cyclic;
    }

    outlet
    {
        type            cyclic;
    }

    walls
    {
        type            noSlip;
    }
}
// ************************************************************************* //""")
    file.close()

def writeFoam_TauDNS_DUCT(filename, tau):
    # Writes the tau file, assumes tau is column stacked
    cells = len(tau)
    leftbracket = np.repeat('(',cells)
    rightbracket = np.repeat(')',cells)
    array_write = np.column_stack((leftbracket,tau,rightbracket))
    print('[dataFoam] Writing TauDNS to file '+filename)
    file = open(filename,'w')
    file.write("""/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2006                                 |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volSymmTensorField;
    object      TauDNS;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField nonuniform List<symmTensor>
""")
    file.write(str(cells) + """ (
""")
    np.savetxt(file, array_write,fmt='%s')
    file.write(""");
boundaryField
{
    inlet
    {
        type            cyclic;
    }

    outlet
    {
        type            cyclic;
    }

    ".*"
    {
        type            fixedValue;
        value           uniform (0 0 0 0 0 0);
    }
}
    // ************************************************************************* //""")
    file.close()

def writeFoam_ap_DUCT(filename, aperp):
    # Writes the aperp file
    cells = len(aperp)
    aperp = np.column_stack((aperp[:,0,0],aperp[:,0,1],aperp[:,0,2],
                                       aperp[:,1,1],aperp[:,1,2],
                                                    -aperp[:,0,0]-aperp[:,1,1]))
    leftbracket = np.repeat('(',len(aperp))
    rightbracket = np.repeat(')',len(aperp))
    array_write = np.column_stack((leftbracket,aperp,rightbracket))
    print('[dataFoam] Writing aperp to file '+filename)
    file = open(filename,'w')
    file.write("""/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2006                                 |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volSymmTensorField;
    object      aperp;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField nonuniform List<symmTensor>
""")
    file.write(str(cells) + """ (
""")
    np.savetxt(file, array_write,fmt='%s')
    file.write(""");
boundaryField
{
    inlet
    {
        type            cyclic;
    }

    outlet
    {
        type            cyclic;
    }

    ".*"
    {
        type            fixedValue;
        value           uniform (0 0 0 0 0 0);
    }
}
// ************************************************************************* //""")
    file.close()

def writeFoam_nut_L_DUCT(filename,nut_L):
    # Writes the nut_L file
    cells = len(nut_L)
    print('[dataFoam] Writing nut_L to file {}'.format(filename))
    file = open(filename,'w')
    file.write("""/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2006                                 |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      nut_L;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -1 0 0 0 0];

internalField nonuniform List<scalar>
""")
    file.write(str(cells) + """ (
""")
    np.savetxt(file, nut_L, fmt='%s')
    file.write(""");
boundaryField
{
    inlet
    {
        type            cyclic;
    }

    outlet
    {
        type            cyclic;
    }

    ".*"
    {
        type            fixedValue;
        value           uniform 0;
    }
}
// ************************************************************************* //""")
    file.close()

def writeFoam_genericscalar_DUCT(filename,field_name,field):
    # Writes the nut_L file
    cells = len(field)
    print('[dataFoam] Writing {} to file {}'.format(field_name,filename))
    file = open(filename,'w')
    file.write("""/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2006                                 |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      """+field_name+""";
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 0 0 0 0 0 0];

internalField nonuniform List<scalar>
""")
    file.write(str(cells) + """ (
""")
    np.savetxt(file, field, fmt='%s')
    file.write(""");
boundaryField
{
    inlet
    {
        type            cyclic;
    }

    outlet
    {
        type            cyclic;
    }

    ".*"
    {
        type            fixedValue;
        value           uniform 0;
    }
}
// ************************************************************************* //""")
    file.close()
