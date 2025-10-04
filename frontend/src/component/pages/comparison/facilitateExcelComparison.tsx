import React from "react";
import "./facilitateComparison.css";

interface FacilitateComparisonProps {
  tableData: any;
}

const FacilitateComparison: React.FC<FacilitateComparisonProps> = ({
  tableData,
}) => {
  const rows = tableData?.data;

  // Generate headers, with "Sl No" as the first column
  const headers = rows?.length > 0 ? ["Sl No", ...Object.keys(rows[0])] : [];

  // Render the table
  return (
    <div
      className="fac-tableContainer"
      style={{ width: "100%", height: "500px" }}
    >
      <div className="fac-tableWrapper">
        <table className="fac-table">
          <thead>
            <tr>
              {headers.map((header, index) => (
                <th key={index}>{header}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows?.map((row: any, rowIndex: number) => (
              <tr key={rowIndex}>
                <td>{rowIndex + 1}</td> {/* Add Sl No */}
                {headers.slice(1).map((header, cellIndex) => (
                  <td key={cellIndex}>
                    {Number.isNaN(row[header]) ? "" : row[header] || ""}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default FacilitateComparison;
