// commercial.tsx
import React from "react";
import "./commercial.css";
import entities from "component/content/content.json";
import LeftArrowIcon from "assets/icon/ip-leftArrow-icon";
import DownloadIcon from "assets/icon/ip-download-icon";

interface CommercialProps {
  onBackClick?: () => void;
}

const Commercial: React.FC<CommercialProps> = ({ onBackClick }) => {
  const handleDownloadClick = () => {};

  return (
    <div className="tableBox">
      <div className="buttons">
        <button
          className="commercialBackButton"
          // onClick={onBackClick}
        >
          <LeftArrowIcon
            className="commercialBackIcon"
            // name="abb/left-arrow"
            size="medium"
          />
          Back
        </button>

        <button className="redClauseButton">Red Clause</button>

        <button className="nonRedClauseButton">Non - red Clause</button>

        <button className="downloadButton" onClick={handleDownloadClick}>
          <DownloadIcon
            className="downloadBackIcon"
            // name="abb/download"
            size="medium"
          />
          Download
        </button>
      </div>

      <table className="table">
        <thead>
          <tr>
            <th className="sno">S.No.</th>
            <th className="entity">Entity/Suplliers</th>
            <th>Supplier 1</th>
            <th>Supplier 2</th>
            <th>Supplier 3</th>
          </tr>
        </thead>
        <tbody>
          {entities.entities.map((entity, index) => (
            <tr key={index}>
              <td className="sno">{index + 1}</td>
              <td className="entity">{entity}</td>
              <td>Row {index + 1} - Data 2</td>
              <td>Row {index + 1} - Data 3</td>
              <td>Row {index + 1} - Data 4</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default Commercial;
