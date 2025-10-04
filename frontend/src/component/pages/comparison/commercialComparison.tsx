import React from "react";
import "./commercialComparison.css";

interface CommercialProps {
  headers: string[];
  data: any[];
}

const CommercialComparison: React.FC<CommercialProps> = ({ headers, data }) => {
  return (
    <>
      {headers.length > 0 && (
        <div className="comp-tableContainer">
          <div className="table-Wrapper">
            <table className="comp-table">
              <thead>
                <tr>
                  <th>Sl. No</th>
                  {headers.map((header, index) => (
                    <th key={index}>{header}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.map((row, rowIndex) => (
                  <tr key={rowIndex}>
                    <td>{rowIndex + 1}</td> {/* Add serial number */}
                    {headers.map((header, colIndex) => (
                      <td key={colIndex}>{row[header]}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  );
};

export default CommercialComparison;
