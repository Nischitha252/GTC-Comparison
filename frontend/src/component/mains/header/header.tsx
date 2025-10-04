//header.tsx
import React, { useState } from "react";
import "./header.css"; // Make sure to create a corresponding CSS file for styling
import logoImage from "../../../assets/abb.png";
import Authentication from "./authentication";
import { useNavigate } from "react-router-dom";

const Header: React.FC = () => {
  const navigate = useNavigate();

  const handleLogoClick = () => {
    navigate("/");
    window.location.reload(); // Refresh the webpage
  };

  return (
    <div>
      <header className="header">
        <div style={{ display: "flex" }} className="logo-cnt">
          <img
            src={logoImage}
            alt="Logo"
            className="logoImage"
            onClick={handleLogoClick}
          />

          <h3 className="header-content">
            {" "}
            <div className="divider"></div>Indirect Procurement{" "}
          </h3>
        </div>
        <div>
          <Authentication />
        </div>
      </header>
    </div>
  );
};

export default Header;
