#pylint: disable=line-too-long
"""Template HEK xml"""

CH_XML = \
'''<?xml version="1.0" encoding="UTF-8"?>
<voe:VOEvent ivorn="ivo://helio-informatics.org/$ivorn" role="observation" version="1.1" xsi:schemaLocation="http://www.ivoa.net/xml/VOEvent/v1.1 http://www.lmsal.com/helio-informatics/VOEvent-v1.1.xsd" xmlns:voe="http://www.ivoa.net/xml/VOEvent/v1.1" xmlns:stc="http://www.ivoa.net/xml/STC/stc-v1.30.xsd" xmlns:lmsal="http://www.lmsal.com/helio-informatics/lmsal-v1.0.xsd" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
      <Who>
        <!--Data pertaining to curation-->
        <AuthorIVORN>ivo://helio-informatics.org/$ivorn</AuthorIVORN>
        <Author>
          <contactName>$author_email</contactName>
        </Author>
        <Date>$datetime</Date>
      </Who>
      <What>
        <!--Data about what was measured/observed.-->
        <Description/>
        <Group name="CoronalHole_optional">
          <Param name="AREA_RAW" value="$area"/>
          <Param name="AREA_UNIT" value="$area_unit"/>
        </Group>
      </What>
      <WhereWhen>
        <!--Data pertaining to when and where something occured-->
        <ObsDataLocation xmlns="http://www.ivoa.net/xml/STC/stc-v1.30.xsd">
          <ObservatoryLocation>
            <AstroCoordSystem/>
            <AstroCoords id="UTC-HPC-TOPO" coord_system_id="UTC-HPC-TOPO"/>
          </ObservatoryLocation>
          <ObservationLocation id="SDO">
            <AstroCoordSystem/>
            <AstroCoords coord_system_id="UTC-HPC-TOPO">
              <Time>
                <TimeInstant>
                  <ISOTime>$datetime</ISOTime>
                </TimeInstant>
              </Time>
              <Position2D unit="arcsec,arcsec">
                <Value2>
                  <C1>$lat_mean</C1>
                  <C2>$long_mean</C2>
                </Value2>
                <Error2>
                  <C1>0</C1>
                  <C2>0</C2>
                </Error2>
              </Position2D>
            </AstroCoords>
            <AstroCoordArea coord_system_id="UTC-HPC-TOPO">
              <TimeInterval>
                <StartTime>
                  <ISOTime>$start_datetime</ISOTime>
                </StartTime>
                <StopTime>
                  <ISOTime>$stop_datetime</ISOTime>
                </StopTime>
              </TimeInterval>
              <Box2>
                <Center>
                  <C1>$bbox_lat</C1>
                  <C2>$bbox_long</C2>
                </Center>
                <Size>
                  <C1>$bbox_lat_size</C1>
                  <C2>$bbox_long_size</C2>
                </Size>
              </Box2>
            </AstroCoordArea>
          </ObservationLocation>
        </ObsDataLocation>
        <Group name="CoronalHole_optional">
          <Param name="BOUND_CCNSTEPS" value="$chaincode_size"/>
          <Param name="BOUND_CCSTARTC1" value="$chaincode_lat_start"/>
          <Param name="BOUND_CCSTARTC2" value="$chaincode_long_start"/>
          <Param name="BOUND_CHAINCODE" value="$chaincode"/>
          <Param name="CHAINCODETYPE" value="ordered list of points in HPC"/>
        </Group>
      </WhereWhen>
      <How>
        <!--Data pertaining to how the feature/event detection was performed-->
        <lmsal:data>
          <lmsal:OBS_ChannelID>AIA 193</lmsal:OBS_ChannelID>
          <lmsal:OBS_Instrument>AIA</lmsal:OBS_Instrument>
          <lmsal:OBS_MeanWavel>193.000</lmsal:OBS_MeanWavel>
          <lmsal:OBS_WavelUnit>Angstroms</lmsal:OBS_WavelUnit>
        </lmsal:data>
        <lmsal:method>
          <lmsal:FRM_Contact>$author_email</lmsal:FRM_Contact>
          <lmsal:FRM_DateRun>$datetime_now</lmsal:FRM_DateRun>
          <lmsal:FRM_HumanFlag>F</lmsal:FRM_HumanFlag>
          <lmsal:FRM_Identifier>$author_id</lmsal:FRM_Identifier>
          <lmsal:FRM_Institute>$author_institute</lmsal:FRM_Institute>
          <lmsal:FRM_Name>$method_name</lmsal:FRM_Name>
          <lmsal:FRM_ParamSet>$param_set</lmsal:FRM_ParamSet>
        </lmsal:method>
        <Group name="CoronalHole_optional">
          <Param name="OBS_DATAPREPURL" value="$prep_url"/>
        </Group>
      </How>
      <Why>
        <Concept>CoronalHole</Concept>
        <lmsal:EVENT_TYPE>CH: CoronalHole</lmsal:EVENT_TYPE>
      </Why>
      <Reference name="FRM_URL" uri="$doi_url"/>
      <Reference name="OBS_DATAPREPURL" uri="$prep_url"/>
      <Reference name="Edge" type="follows" uri="ivo://helio-informatics.org/$ivorn"/>
    </voe:VOEvent>
'''
