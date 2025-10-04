import React from "react";
import { Size, getSizeDimensions } from "./iconUtils";

interface InfoIconProps {
  className?: string; // Optional className prop
  size?: Size; // Optional size prop
}

const InfoIcon: React.FC<InfoIconProps> = ({ className, size = "medium" }) => {
  const { width, height } = getSizeDimensions(size);

  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 32 32"
      width={width}
      height={height}
      className={className}
    >
      <defs>
        <style>{`.cls-1 { fill: none; } .cls-2 { fill: #1f1f1f; }`}</style>
      </defs>

      <title>abb_information-circle-1_32</title>

      <g id="Box">
        <rect className="cls-1" width="32" height="32" />
      </g>
      <g id="Final_icons_-_Common" data-name="Final icons - Common">
        <path
          className="cls-2"
          d="M16,2A14,14,0,1,0,30,16,14,14,0,0,0,16,2Zm2,23H15V16H13V14h3.36A1.65,1.65,0,0,1,18,15.64ZM16,12a2,2,0,1,1,2-2A2,2,0,0,1,16,12Z"
        />
        <path
          className="cls-2"
          d="M16,2A14,14,0,1,0,30,16,14,14,0,0,0,16,2Zm2,23H15V16H13V14h3.36A1.65,1.65,0,0,1,18,15.64ZM16,12a2,2,0,1,1,2-2A2,2,0,0,1,16,12Z"
        />
      </g>
    </svg>
  );
};

export default InfoIcon;
