// ip-gtc-icon.tsx
import React from 'react';
import {Size, getSizeDimensions} from './iconUtils';

interface ContractsIconProps {
    className?: string; // Optional className prop
    size?: Size; // Optional size prop
}

const ContractsIcon: React.FC<ContractsIconProps> = ({  className, size = "medium" }) => {
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
                <style>
                    {`.contracts-cls-1 { fill: none; } .contracts-cls-2 { fill: currentColor; }`} {/* Define your styles here */}
                </style>
            </defs>

            <title>IP GTC</title>

            <g id="Box">
                <rect className="contracts-cls-1" x="1" width="32" height="32" />
            </g>
            
            <g id="Robotics">
                <rect className="contracts-cls-2" x="14" y="10" width="6" height="2" />
                <path className="contracts-cls-2" d="M21.67,2H9V27H30V10.37ZM28,24v1.09H11V4h9.92L28,11.28Z" />
                <rect className="contracts-cls-2" x="14" y="18" width="11" height="2" />
                <rect className="contracts-cls-2" x="14" y="14" width="11" height="2" />
                <polygon className="contracts-cls-2" points="7 28 7 27 7 8 8 8 8 6 5 6 5 27 5 28 5 30 7 30 26 30 28 30 28 28 26 28 7 28" />
            </g>
        </svg>
    );
};

export default ContractsIcon;
