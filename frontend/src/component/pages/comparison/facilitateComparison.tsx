import React from "react";
import "./facilitateComparison.css";
import FacilitiesExcelComparison from "./facilitateExcelComparison";

interface FacilitateComparisonProps {
  tableData: any;
}

const FacilitateComparison: React.FC<FacilitateComparisonProps> = ({
  tableData,
}) => {
  let parsedTableData;
  try {
    parsedTableData =
      typeof tableData === "string"
        ? JSON.parse(tableData.replace(/NaN/g, "null"))
        : tableData;
  } catch (error) {
    console.error("Error parsing tableData:", error);
    parsedTableData = null;
  }

  // Check if tableData.data is available; if not, render FacilitiesExcelComparison
  if (parsedTableData?.data) {
    return <FacilitiesExcelComparison tableData={parsedTableData} />;
  }

  if (!tableData || typeof tableData !== "object") {
    return null;
  }

  const prompt_details = tableData.prompt_details || {};
  const results = tableData.results || {};
  const pdfFiles = Object.keys(results);

  // Function to extract unique properties for each PDF
  const getPDFPropertyMap = (results: any) => {
    const pdfPropertyMap: { [key: string]: Set<string> } = {};

    // Iterate through each PDF in results
    Object.entries(results).forEach(([pdfName, pdfData]: [string, any]) => {
      pdfPropertyMap[pdfName] = new Set();

      // Check all prompts in the PDF
      if (pdfData.prompts) {
        Object.values(pdfData.prompts).forEach((prompt: any) => {
          // Add all properties found in this prompt
          Object.keys(prompt).forEach((prop) => {
            pdfPropertyMap[pdfName].add(prop);
          });
        });
      }
    });

    // Convert Sets to arrays
    return Object.fromEntries(
      Object.entries(pdfPropertyMap).map(([pdf, propSet]) => [
        pdf,
        Array.from(propSet),
      ])
    );
  };

  // Get PDF property mapping
  const pdfPropertyMap = getPDFPropertyMap(results);

  // Create headers
  const headers = ["Sl No", "Detail"];
  Object.entries(pdfPropertyMap).forEach(([pdf, properties]) => {
    properties.forEach((prop) => {
      headers.push(`${pdf} - ${prop}`);
    });
  });

  // Create rows
  const rows = Object.keys(prompt_details).map((key, index) => {
    const row: any = {
      "Sl No": index + 1,
      Detail: prompt_details[key] || "",
    };

    // Add values only for properties that belong to each PDF
    Object.entries(pdfPropertyMap).forEach(([pdf, properties]) => {
      const pdfData = results[pdf]?.prompts?.[key] || {};
      properties.forEach((prop) => {
        row[`${pdf} - ${prop}`] = pdfData[prop] || "";
      });
    });

    return row;
  });

  if (rows.length === 0) {
    return null;
  }

  return (
    <div
      className="fac-tableContainer"
      style={{ height: headers.length > 0 ? "500px" : "", width: "100%" }}
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
            {rows.map((row: any, index: number) => (
              <tr key={index}>
                {headers.map((header, i) => (
                  <td key={i}>{row[header] || ""}</td>
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
