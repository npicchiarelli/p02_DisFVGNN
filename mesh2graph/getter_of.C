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

class getter_of
{
public:
autoPtr<argList> _args;
autoPtr<fvMesh> _mesh;
autoPtr<Foam::Time> _runTime; 
    getter_of(int argc, char* argv[])
    {
        _args = autoPtr<argList>(
            new argList(argc, argv, true, true, /*initialise=*/false));
        argList& args = _args();
        #include "createTime.H"
        #include "createMesh.H"
    }

    // Getter for face surface area vectors

    std::vector<Eigen::MatrixXd> getSfByPatch()
    {
        std::vector<Eigen::MatrixXd> result;

        vectorField internalSf(_mesh().Sf().internalField());
        result.push_back(Foam2Eigen::field2Eigen(internalSf));

        for (int i = 0; i < _mesh().Sf().boundaryField().size(); i++)
        {
            vectorField X(_mesh().Sf().boundaryField()[i]);
            result.push_back(Foam2Eigen::field2Eigen(X));
        }
        return result;
    }

    Eigen::MatrixXd getSf()
    {
        // Full face area vector field (internal + boundary faces)
        const vectorField& Sf = _mesh().Sf();
        return Foam2Eigen::field2Eigen(Sf);
    }

    // Getter for boundary face centers

    std::vector<Eigen::MatrixXd> getCf()
    {
        std::vector<Eigen::MatrixXd> result;
        for (int i = 0; i < _mesh().Cf().boundaryField().size(); i++)
        {
            vectorField X(_mesh().Cf().boundaryField()[i]);
            Eigen::MatrixXd Teig(Foam2Eigen::field2Eigen(X));
            result.push_back(Teig);
        }
        return result;
    }

    // Getter for boundary patch names

    std::vector<std::string> getPatchName()
    {
        std::vector<std::string> result;
        for (int i = 0; i < _mesh().boundary().size(); i++)
        {
            const word& patchName = _mesh().boundary()[i].name();
            result.push_back(patchName);
        }
        return result;
    }

    Eigen::VectorXd getSkewness()
    {
        // skewness() returns values for all faces including boundary
        const scalarField skewness = primitiveMeshTools::faceSkewness(
            _mesh(),
            _mesh().points(),
            _mesh().faceCentres(),
            _mesh().faceAreas(),
            _mesh().cellCentres()
        );
        return Foam2Eigen::field2Eigen(skewness);
    }

    Eigen::VectorXd getNonOrthogonality()
    {
        const fvMesh& mesh = _mesh();
        const labelList& own = mesh.faceOwner();
        const labelList& nei = mesh.faceNeighbour();
        const vectorField& cc = mesh.cellCentres();
        const vectorField& areas = mesh.faceAreas();

        scalarField nonOrtho(mesh.nFaces(), 0.0);

        // Internal faces
        for (label facei = 0; facei < mesh.nInternalFaces(); facei++)
        {
            nonOrtho[facei] = primitiveMeshTools::faceOrthogonality(
                cc[own[facei]],
                cc[nei[facei]],
                areas[facei]
            );
        }

        // Boundary faces: no neighbour cell, set to 1 (perfectly orthogonal)
        for (label facei = mesh.nInternalFaces(); facei < mesh.nFaces(); facei++)
        {
            nonOrtho[facei] = 1.0;
        }

        return Foam2Eigen::field2Eigen(nonOrtho);
    }
};


PYBIND11_MODULE(getter_of, m)
{
    // bindings to Matrix class
    py::class_<getter_of>(m, "getter_of")
        .def(py::init([](
                          std::vector<std::string> args) {
            std::vector<char*> cstrs;
            cstrs.reserve(args.size());
            for (auto& s : args)
                cstrs.push_back(const_cast<char*>(s.c_str()));
            return new getter_of(cstrs.size(), cstrs.data());
        }),
            py::arg("args") = std::vector<std::string> { "." })
            .def("getCf", &getter_of::getCf, py::return_value_policy::reference_internal)
            .def("getPatchName", &getter_of::getPatchName)
            .def("getSf", &getter_of::getSf, py::return_value_policy::reference_internal)
            .def("getSfByPatch", &getter_of::getSfByPatch, py::return_value_policy::reference_internal)
            .def("getSkewness", &getter_of::getSkewness, py::return_value_policy::reference_internal)
            .def("getNonOrthogonality", &getter_of::getNonOrthogonality, py::return_value_policy::reference_internal);
}
