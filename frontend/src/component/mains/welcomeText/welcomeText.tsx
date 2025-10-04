import React from "react";
import "./welcomeText.css";
// Define the props interface
interface WelcomeTextProps {
  isLoading?: boolean;
}

const WelcomeText: React.FC<WelcomeTextProps> = ({ isLoading }) => {
  return (
    <div className="welcomeText">
      <div className="redLine" />
      <h2>Welcome to Indirect Procurement</h2>
      <p>
        Indirect procurement refers to the purchasing of goods and services that
        are not directly involved in the production process of a company's goods
        or services. It deals with everything else a business needs to operate
        efficiently. This can include office supplies, IT services, marketing
        services, travel expenses, and more. Effective management of indirect
        procurement can help streamline their operations, control costs, and
        optimize their overall procurement processes.
      </p>
      {isLoading && (
        <h3>Please click the option(s) below for GTC comparison:</h3>
      )}
    </div>
  );
};

export default WelcomeText;
