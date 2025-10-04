// // authentication.tsx
import React, { useState, useEffect, useRef } from "react";
import { useMsal } from "@azure/msal-react";
import "./authentication.css";
import LogOutIcon from "assets/icon/ip-logout-icon";

const Authentication : React.FC = () => {
    const { instance, accounts } = useMsal();
    const [showDropdown, setShowDropdown] = useState(false);
    const dropdownRef = useRef<HTMLDivElement>(null); // Reference to the dropdown div
    const userDetailsDropdownRef = useRef<HTMLSpanElement>(null); // Reference to the userDetails span

    const userDetails = accounts && accounts[0];
    const { name, username } = userDetails || {};

    useEffect(() => {
        // Function to handle clicks outside of the dropdown
        const handleClickOutside = (event: MouseEvent) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node) && 
                userDetailsDropdownRef.current && !userDetailsDropdownRef.current.contains(event.target as Node)) {
                setShowDropdown(false); // Close dropdown if clicked outside of it
            }
        };

        // Adding event listener for clicks outside of the dropdown
        document.addEventListener("mousedown", handleClickOutside);

        // Cleanup function to remove event listener on component unmount
        return () => {
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, []);

    const handleLogOut = async () => {
        await instance.logoutRedirect();
    };

    const handleRedirectToTerms = () => {
        window.location.href = '/terms-and-conditions'; // Change this to your desired URL
        // history.push('/terms-and-conditions'); // Change this to your desired URL
    };

    return (
        <div>
            {userDetails && (
                <div>
                    <span 
                        className="userDetails"  
                        ref={userDetailsDropdownRef}
                        onClick={() => setShowDropdown(!showDropdown)} 
                    >
                        <span className="userName">{name}</span>
                        <span className="userEmail">{username}</span>
                    </span>

                    {showDropdown && (
                        <div className="AuthDropdown" ref={dropdownRef}>
                            {/* <button className="tAndC" onClick={handleRedirectToTerms}>
                                <ABB.Icon 
                                    className="tAndCIcon"
                                    name="abb/reports" 
                                    sizeClass="medium" 
                                />
                                <span> Terms & Conditions </span>
                            </button> */}

                            <button className="logOutButton" onClick={handleLogOut}>
                                <LogOutIcon className="logOutButtonIcon" size="medium" />
                                <span> Logout </span>
                            </button>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

export default Authentication ;
