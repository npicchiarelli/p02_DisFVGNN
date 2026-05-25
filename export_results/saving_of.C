/*---------------------------------------------------------------------------*\
     ██╗████████╗██╗  ██╗ █████╗  ██████╗ █████╗       ███████╗██╗   ██╗
     ██║╚══██╔══╝██║  ██║██╔══██╗██╔════╝██╔══██╗      ██╔════╝██║   ██║
     ██║   ██║   ███████║███████║██║     ███████║█████╗█████╗  ██║   ██║
     ██║   ██║   ██╔══██║██╔══██║██║     ██╔══██║╚════╝██╔══╝  ╚██╗ ██╔╝
     ██║   ██║   ██║  ██║██║  ██║╚██████╗██║  ██║      ██║      ╚████╔╝
     ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝      ╚═╝       ╚═══╝

 * In real Time Highly Advanced Computational Applications for Finite Volumes
 * Copyright (C) 2017 by the ITHACA-FV authors
-------------------------------------------------------------------------------
License
    This file is part of ITHACA-FV
    ITHACA-FV is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    ITHACA-FV is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with ITHACA-FV. If not, see <http://www.gnu.org/licenses/>.
Description
    Example of a heat transfer Reduction Problem
SourceFiles
    02thermalBlock.C
\*---------------------------------------------------------------------------*/


#include "fvCFD.H"
#include "IOmanip.H"
#include "Time.H"
#include "ReducedLaplacian.H"
#include "ITHACAPOD.H"
#include "ITHACAutilities.H"
#include <cstddef>
#define _USE_MATH_DEFINES
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include "iostream"
#include "Foam2Eigen.H"
#include "primitiveMeshTools.H"
#include "polyMeshTools.H"



#if PY_VERSION_HEX < 0x03000000
#define MyPyText_AsString PyString_AsString
#else
#define MyPyText_AsString PyUnicode_AsUTF8
#endif

namespace py = pybind11;

class saving_of
{
public:
    autoPtr<argList> _args;
    autoPtr<fvMesh> _mesh;
    autoPtr<Foam::Time> _runTime; 
    autoPtr<volScalarField> _SF;

    saving_of(int argc, char* argv[])
    {
        _args = autoPtr<argList>(
            new argList(argc, argv, true, true, /*initialise=*/false));
        argList& args = _args();
        #include "createTime.H"
        #include "createMesh.H"

        // Allocate _SF after mesh is ready
        _SF = autoPtr<volScalarField>
        (
            new volScalarField
            (
                IOobject
                (
                    "SF",
                    _runTime->timeName(),
                    _mesh(),
                    IOobject::NO_READ,
                    IOobject::AUTO_WRITE
                ),
                _mesh(),
                dimensionedScalar("SF", dimless, 0.0)
            )
        );
    }

    void setScalarField(Eigen::VectorXd F)
    {
        _SF() = Foam2Eigen::Eigen2field(_SF(), F);
    }

    void exportScalarField(std::string& subFolder, std::string& folder, std::string& fieldname)
    {
        ITHACAstream::exportSolution(_SF(), subFolder, folder, fieldname);
    }
};


PYBIND11_MODULE(saving_of, m)
{
    // bindings to Matrix class
    py::class_<saving_of>(m, "saving_of")
        .def(py::init([](
                          std::vector<std::string> args) {
            std::vector<char*> cstrs;
            cstrs.reserve(args.size());
            for (auto& s : args)
                cstrs.push_back(const_cast<char*>(s.c_str()));
            return new saving_of(cstrs.size(), cstrs.data());
        }),
            py::arg("args") = std::vector<std::string> { "." })
            .def("setScalarField", &saving_of::setScalarField)
            .def("exportScalarField", &saving_of::exportScalarField);
}
