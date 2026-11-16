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
#include <limits>
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

    // Getter for cell volumes, one per cell in cell index order

    Eigen::VectorXd getV()
    {
        const scalarField& V = _mesh().cellVolumes();
        return Foam2Eigen::field2Eigen(V);
    }

    // ------------------------------------------------------------------------
    // Diffusion operator, for its eigenpairs (mesh2graph/laplacian_eigen.py).
    //
    // Nothing about the physics is hard-coded here; it is read from the case
    // the way laplacianFoam reads it:
    //  - the field and its boundary conditions, from <timeName>/<fieldName>.
    //    Each BC enters through its homogeneous part (the matrix internalCoeffs:
    //    fixedValue -> Dirichlet, zeroGradient/fixedGradient -> Neumann,
    //    mixed -> Robin); the inhomogeneous part (the source) is dropped.
    //  - the diffusivity: the volScalarField <timeName>/<coeffName> if present
    //    (non-uniform), else the uniform <coeffName> entry of
    //    constant/transportProperties.
    //  - the discretisation, from the laplacian(<coeffName>,<fieldName>) entry
    //    of system/fvSchemes, including how the diffusivity is interpolated to
    //    the faces (e.g. Gauss harmonic corrected for one that jumps).
    // The temporaries are not registered on the mesh, so repeated calls do not
    // collide with each other or with fields of the same name.

    // Diffusivity, as laplacianFoam's createFields.H reads it.
    tmp<volScalarField> readDiffusivity(const word& coeffName, const word& timeName)
    {
        const fvMesh& mesh = _mesh();
        IOobject io(coeffName, timeName, mesh,
                    IOobject::MUST_READ, IOobject::NO_WRITE, false);

        // checkType, no search: only <timeName> itself counts
        if (io.typeHeaderOk<volScalarField>(true, false))
        {
            Info<< "getter_of: non-uniform " << coeffName << " from "
                << timeName << "/" << coeffName << endl;
            return tmp<volScalarField>::New(io, mesh);
        }

        IOdictionary transportProperties
        (
            IOobject("transportProperties", _runTime().constant(), mesh,
                     IOobject::MUST_READ, IOobject::NO_WRITE, false)
        );
        // M = diag(V) in the eigenproblem assumes a diffusivity, hence its
        // dimensions are checked here
        const dimensionedScalar uniform(coeffName, dimViscosity, transportProperties);
        Info<< "getter_of: uniform " << coeffName << " = " << uniform.value()
            << " from constant/transportProperties" << endl;

        io.readOpt(IOobject::NO_READ);
        return tmp<volScalarField>::New(io, mesh, uniform);
    }

    volScalarField readField(const word& fieldName, const word& timeName)
    {
        return volScalarField
        (
            IOobject(fieldName, timeName, _mesh(),
                     IOobject::MUST_READ, IOobject::NO_WRITE, false),
            _mesh()
        );
    }

    // K = -fvm::laplacian(DT, T), the homogeneous part only, cells x cells.
    // Symmetric; positive definite if DT > 0 and any patch fixes the value.
    Eigen::SparseMatrix<double> getLaplacian(const std::string& fieldName = "T",
                                             const std::string& coeffName = "DT",
                                             const std::string& timeName = "0")
    {
        const fvMesh& mesh = _mesh();
        const volScalarField T(readField(fieldName, timeName));

        // fvMatrix2Eigen moves the coupling of coupled patches (cyclic,
        // processor, AMI) into the source, which would silently cut it from K
        forAll(T.boundaryField(), patchi)
        {
            if (T.boundaryField()[patchi].coupled())
            {
                FatalErrorInFunction
                    << "coupled patch " << mesh.boundary()[patchi].name()
                    << " is not supported (no cyclics; run serial)"
                    << exit(FatalError);
            }
        }

        tmp<volScalarField> tDT = readDiffusivity(coeffName, timeName);
        const volScalarField& DT = tDT();
        scalar minDT = gMin(DT.primitiveField());
        forAll(DT.boundaryField(), patchi)
        {
            if (DT.boundaryField()[patchi].size())
            {
                minDT = min(minDT, gMin(DT.boundaryField()[patchi]));
            }
        }
        if (minDT <= 0)
        {
            FatalErrorInFunction
                << coeffName << " must be > 0 everywhere, min = " << minDT
                << exit(FatalError);
        }

        // fvSchemes key: laplacian(<coeffName>,<fieldName>), as in the solver
        fvScalarMatrix L(fvm::laplacian(DT, T));

        Eigen::SparseMatrix<double> A;
        Eigen::VectorXd b;
        Foam2Eigen::fvMatrix2Eigen(L, A, b);   // b: inhomogeneous BC part, dropped
        return -A;
    }

    // Per patch, in getPatchName order, sized as the polyPatch: c such that the
    // homogeneous BC gives phi_face = c * phi_owner (fixedValue 0, zeroGradient
    // 1, mixed 1 - valueFraction). NaN on patches without face values (empty).
    std::vector<Eigen::VectorXd> getBoundaryValueCoeffs(const std::string& fieldName = "T",
                                                        const std::string& timeName = "0")
    {
        const fvMesh& mesh = _mesh();
        const volScalarField T(readField(fieldName, timeName));

        std::vector<Eigen::VectorXd> result;
        forAll(T.boundaryField(), patchi)
        {
            const label nFaces = mesh.boundaryMesh()[patchi].size();
            Eigen::VectorXd c = Eigen::VectorXd::Constant(
                nFaces, std::numeric_limits<double>::quiet_NaN());

            const fvPatchScalarField& pf = T.boundaryField()[patchi];
            if (pf.size() == nFaces && nFaces > 0)
            {
                const scalarField coeffs(
                    pf.valueInternalCoeffs(mesh.boundary()[patchi].weights()));
                c = Foam2Eigen::field2Eigen(coeffs);
            }
            result.push_back(c);
        }
        return result;
    }

    // Diffusivity per cell, read as getLaplacian reads it.
    Eigen::VectorXd getDiffusivity(const std::string& coeffName = "DT",
                                   const std::string& timeName = "0")
    {
        tmp<volScalarField> tDT = readDiffusivity(coeffName, timeName);
        return Foam2Eigen::field2Eigen(tDT().primitiveField());
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
            .def("getNonOrthogonality", &getter_of::getNonOrthogonality, py::return_value_policy::reference_internal)
            .def("getV", &getter_of::getV)
            .def("getLaplacian", &getter_of::getLaplacian,
                 py::arg("fieldName") = "T", py::arg("coeffName") = "DT",
                 py::arg("timeName") = "0")
            .def("getBoundaryValueCoeffs", &getter_of::getBoundaryValueCoeffs,
                 py::arg("fieldName") = "T", py::arg("timeName") = "0")
            .def("getDiffusivity", &getter_of::getDiffusivity,
                 py::arg("coeffName") = "DT", py::arg("timeName") = "0");
}
